# SVModeller - Module 2 (inversiones)
#
# Input:
# - Inversion_Lengths.tsv (salida del Module 1): tamaños observados de las inversiones
# - Tabla de regiones invertibles: bp1_1, bp1_2, bp2_1, bp2_2, region (p. ej. "chr1:1587399-1590825")
#
# Output:
# - Inversions_table.tsv : inversiones simuladas con sus breakpoints
# - VCF_Inversions_SVModeller.vcf : VCF con las inversiones simuladas (opcional con --VCF)
# - Fitted_Distribution.txt : distribución ajustada a los tamaños observados
#
# Procedimiento:
# 1. Se ajustan distribuciones estadísticas (distfit, 89 distribuciones) a los tamaños observados
#    y se elige la de menor RSS (mismo criterio que el Module 2 de inserciones).
# 2. Se generan tamaños aleatorios con esa distribución (se descartan valores <= 0).
# 3. Para cada tamaño simulado se buscan las regiones invertibles cuyo segmento interno
#    (bp2_1 - bp1_2) difiera como máximo un 10% del tamaño simulado, y se elige una al azar.
#    Los breakpoints de la inversión son los de la región elegida:
#    BP1 = bp1_1-bp1_2 y BP2 = bp2_1-bp2_2.
# 4. Si una región solapa con otra ya elegida, se descarta (se conserva solo una de ellas).

import argparse
import datetime
import os
import random
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

EXCLUDED_CHROMS = {'chrY', 'Y'}  # no se simulan inversiones en chrY


def set_seed(seed: int = 42):
    """Fija las semillas aleatorias para que los resultados sean reproducibles."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_lengths(path):
    """Lee los tamaños observados (columna 'Length' de Inversion_Lengths.tsv)."""
    df = pd.read_csv(path, sep='\t')
    col = 'Length' if 'Length' in df.columns else df.columns[0]
    lengths = pd.to_numeric(df[col], errors='coerce').dropna()
    lengths = lengths[lengths > 0].astype(int).to_numpy()
    if len(lengths) < 3:
        raise ValueError('Se necesitan al menos 3 tamaños observados para ajustar una distribución.')
    print(f'Tamaños observados: {len(lengths)}')
    return lengths


def read_regions(path):
    """
    Lee la tabla de regiones invertibles (tabulador o espacios, con cabecera):
    bp1_1, bp1_2, bp2_1, bp2_2, region
    El cromosoma se obtiene de la columna 'region' (p. ej. chr1:1587399-1590825).
    """
    df = pd.read_csv(path, sep=r'\s+', engine='python')
    df.columns = [c.strip() for c in df.columns]
    required = ['bp1_1', 'bp1_2', 'bp2_1', 'bp2_2', 'region']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas en el fichero de regiones: {missing}')

    for col in ['bp1_1', 'bp1_2', 'bp2_1', 'bp2_2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['chr'] = df['region'].astype(str).str.split(':').str[0]
    df['inner'] = df['bp2_1'] - df['bp1_2']  # segmento interno = tamaño de la inversión

    n0 = len(df)
    invalid = df[['bp1_1', 'bp1_2', 'bp2_1', 'bp2_2']].isna().any(axis=1) | (df['inner'] <= 0)
    if invalid.any():
        print(f'Aviso: se descartan {invalid.sum()} regiones con coordenadas inválidas o segmento interno <= 0')
    df = df[~invalid]

    y = df['chr'].isin(EXCLUDED_CHROMS)
    if y.any():
        print(f'Aviso: se descartan {y.sum()} regiones en chrY')
    df = df[~y].copy()

    for col in ['bp1_1', 'bp1_2', 'bp2_1', 'bp2_2', 'inner']:
        df[col] = df[col].astype(np.int64)
    df = df.reset_index(drop=True)
    print(f'Regiones invertibles: {len(df)} (de {n0})')
    return df


def fit_size_distribution(lengths, distr='full'):
    """
    Ajusta distribuciones con distfit y selecciona la de menor RSS.
    distr='full' evalúa las 89 distribuciones disponibles.
    """
    from distfit import distfit
    dist = distfit(distr=distr, stats='RSS')
    dist.fit_transform(lengths, verbose=0)
    return dist


def write_fit_summary(dist, lengths, out_file='Fitted_Distribution.txt'):
    model = getattr(dist, 'model', {}) or {}
    name = model.get('name', 'NA')
    score = model.get('score', 'NA')
    params = model.get('params', 'NA')
    with open(out_file, 'w') as f:
        f.write(f'N_observed\t{len(lengths)}\n')
        f.write(f'Best_distribution\t{name}\n')
        f.write(f'RSS\t{score}\n')
        f.write(f'Parameters\t{params}\n')
    print(f'Distribución seleccionada (menor RSS): {name}')


def generate_sizes(dist, n):
    """Genera n tamaños aleatorios con la distribución ajustada y elimina valores no válidos (<= 0)."""
    sizes = np.asarray(dist.generate(n), dtype=float)
    sizes = sizes[np.isfinite(sizes)]
    sizes = np.rint(sizes).astype(np.int64)
    return sizes[sizes > 0]  # equivalente a remove_negative_values()


def simulate_inversions(regions, sizes, num_inversions, tolerance):
    """
    Para cada tamaño simulado elige al azar una región cuyo segmento interno esté dentro de
    ±tolerance (relativo al tamaño simulado). Al aceptar una región, se marcan como no disponibles
    todas las regiones que solapen con ella (sobre bp1_1 - bp2_2, en el mismo cromosoma).
    """
    chrom = regions['chr'].to_numpy()
    beg = regions['bp1_1'].to_numpy()
    end = regions['bp2_2'].to_numpy()
    inner = regions['inner'].to_numpy()
    available = np.ones(len(regions), dtype=bool)

    chosen = []  # (índice de región, tamaño simulado)
    attempts = 0
    no_candidate = 0

    for size in sizes:
        if len(chosen) >= num_inversions or not available.any():
            break
        attempts += 1
        candidates = np.flatnonzero(available & (np.abs(inner - size) <= tolerance * size))
        if candidates.size == 0:
            no_candidate += 1
            continue
        idx = int(np.random.choice(candidates))
        chosen.append((idx, int(size)))
        overlap = (chrom == chrom[idx]) & (beg <= end[idx]) & (end >= beg[idx])
        available &= ~overlap

    print(f'Intentos: {attempts} | sin región compatible: {no_candidate} | inversiones simuladas: {len(chosen)}')
    if len(chosen) < num_inversions:
        print(f'Aviso: solo se pudieron simular {len(chosen)} de {num_inversions} inversiones '
              f'(sin solapamiento y con la tolerancia de tamaño indicada).')

    rows = []
    for idx, size in chosen:
        r = regions.iloc[idx]
        rows.append({
            'chr': r['chr'], 'bp1_1': r['bp1_1'], 'bp1_2': r['bp1_2'],
            'bp2_1': r['bp2_1'], 'bp2_2': r['bp2_2'],
            'Sim_size': size, 'Length': int(r['inner']),
            'Size_diff_pct': round(100 * (int(r['inner']) - size) / size, 2),
            'region': r['region'],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table['_k'] = table['chr'].map(chrom_key)
    table = table.sort_values(['_k', 'bp1_2']).drop(columns='_k').reset_index(drop=True)
    table.insert(0, 'ID', [f'SIMINV_{i + 1:04d}' for i in range(len(table))])
    return table


def chrom_key(c):
    c = str(c).replace('chr', '')
    if c.isdigit():
        return (0, int(c))
    return (1, {'X': 23, 'Y': 24, 'M': 25, 'MT': 25}.get(c, 99))


def create_vcf_file(table, out_file='VCF_Inversions_SVModeller.vcf', reference_fasta=None):
    """
    VCF con las inversiones simuladas:
    POS = bp1_2, END = bp2_1 (segmento invertido), SVLEN = END - POS
    CIPOS = (bp1_1 - POS, 0) y CIEND = (0, bp2_2 - END) describen los intervalos de breakpoint.
    """
    fasta = None
    if reference_fasta:
        import pysam
        fasta = pysam.FastaFile(reference_fasta)

    with open(out_file, 'w') as f:
        f.write('##fileformat=VCFv4.2\n')
        f.write(f'##fileDate={datetime.date.today().strftime("%Y%m%d")}\n')
        f.write('##source=SVModeller_Inversions\n')
        if reference_fasta:
            f.write(f'##reference={os.path.basename(reference_fasta)}\n')
        for c in sorted(table['chr'].unique(), key=chrom_key):
            f.write(f'##contig=<ID={c}>\n')
        f.write('##ALT=<ID=INV,Description="Inversion">\n')
        f.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
        f.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the inverted segment (bp2_1)">\n')
        f.write('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="Length of the inverted segment (bp2_1 - bp1_2)">\n')
        f.write('##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="Confidence interval around POS (breakpoint 1: bp1_1-bp1_2)">\n')
        f.write('##INFO=<ID=CIEND,Number=2,Type=Integer,Description="Confidence interval around END (breakpoint 2: bp2_1-bp2_2)">\n')
        f.write('##INFO=<ID=SIM_SIZE,Number=1,Type=Integer,Description="Size drawn from the fitted distribution before fitting to the region">\n')
        f.write('##INFO=<ID=REGION,Number=1,Type=String,Description="Invertible region used">\n')
        f.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')

        for _, r in table.iterrows():
            pos, end = int(r['bp1_2']), int(r['bp2_1'])
            ref = 'N'
            if fasta is not None:
                ref = fasta.fetch(r['chr'], pos - 1, pos).upper() or 'N'
            info = (f"SVTYPE=INV;END={end};SVLEN={end - pos};"
                    f"CIPOS={int(r['bp1_1']) - pos},0;CIEND=0,{int(r['bp2_2']) - end};"
                    f"SIM_SIZE={int(r['Sim_size'])};REGION={r['region']}")
            f.write(f"{r['chr']}\t{pos}\t{r['ID']}\t{ref}\t<INV>\t.\tPASS\t{info}\n")

    if fasta is not None:
        fasta.close()
    print(f'VCF creado: {out_file}')


def main(lengths_path, regions_path, num_inversions, tolerance, distr, max_attempts,
         apply_VCF, reference_fasta, seed):
    print(f'Fichero de tamaños: {lengths_path}')
    print(f'Fichero de regiones invertibles: {regions_path}')
    print(f'Inversiones a simular: {num_inversions} | tolerancia de tamaño: ±{tolerance * 100:.0f}%')

    set_seed(seed)
    lengths = read_lengths(lengths_path)
    regions = read_regions(regions_path)

    dist = fit_size_distribution(lengths, distr)
    write_fit_summary(dist, lengths)

    sizes = generate_sizes(dist, max_attempts)
    table = simulate_inversions(regions, sizes, num_inversions, tolerance)

    table.to_csv('Inversions_table.tsv', sep='\t', index=False)
    print('Tabla guardada: Inversions_table.tsv')

    if apply_VCF and not table.empty:
        create_vcf_file(table, reference_fasta=reference_fasta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simula inversiones a partir de la distribución de tamaños observada y de regiones invertibles.')
    parser.add_argument('--lengths_path', type=str, required=True, help='Inversion_Lengths.tsv generado por el Module 1.')
    parser.add_argument('--regions_path', type=str, required=True, help='Tabla de regiones invertibles (bp1_1, bp1_2, bp2_1, bp2_2, region).')
    parser.add_argument('--num_inversions', type=int, default=1000, help='Número de inversiones a simular (default: 1000).')
    parser.add_argument('--tolerance', type=float, default=0.10, help='Variación de tamaño permitida, en fracción (default: 0.10 = 10%%).')
    parser.add_argument('--distr', type=str, default='full', help="Conjunto de distribuciones de distfit: 'full' (89), 'popular'... (default: full).")
    parser.add_argument('--max_attempts', type=int, default=100000, help='Máximo de tamaños simulados que se intentan colocar (default: 100000).')
    parser.add_argument('--VCF', action='store_true', help='Crea también un VCF con las inversiones simuladas.')
    parser.add_argument('--reference_fasta', type=str, default=None, help='Genoma de referencia (opcional, para poner la base REF real en el VCF; requiere pysam).')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria (default: 42).')
    args = parser.parse_args()
    main(args.lengths_path, args.regions_path, args.num_inversions, args.tolerance, args.distr,
         args.max_attempts, args.VCF, args.reference_fasta, args.seed)
