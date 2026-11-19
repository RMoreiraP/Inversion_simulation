import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

# Cromosomas para la distribución genómica (chrY excluido: no se simula ahí)
CHROMOSOMES = [f'chr{i}' for i in range(1, 23)] + ['chrX']


def set_seed(seed: int = 42):
    """Fija las semillas aleatorias para que los resultados sean reproducibles."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_inversion_table(file_path):
    """
    Lee la tabla de inversiones (columnas separadas por tabulador o espacios):
    inversion, chr, bp1_1, bp1_2, bp2_1, bp2_2, Type, Mediating, Global freq
    """
    # Se leen solo las 8 primeras columnas por posición (ignora "Global freq", que lleva un espacio en la cabecera)
    names = ['inversion', 'chr', 'bp1_1', 'bp1_2', 'bp2_1', 'bp2_2', 'Type', 'Mediating']
    df = pd.read_csv(file_path, sep=r'\s+', engine='python', header=0, names=names, usecols=range(8))

    for col in ['bp1_1', 'bp1_2', 'bp2_1', 'bp2_2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def process_inversion_table(df):
    """
    - Se queda solo con las inversiones NAHR
    - Calcula Length = bp2_1 - bp1_2 (segmento interno)
    - Descarta (y avisa de) las de tamaño <= 0 o con coordenadas inválidas
    - Ignora la frecuencia global
    """
    # Solo NAHR
    n_total = len(df)
    df = df[df['Type'] == 'NAHR'].copy()
    print(f'Inversiones totales: {n_total} | NAHR: {len(df)}')

    # Tamaño = segmento interno
    df['Length'] = df['bp2_1'] - df['bp1_2']

    # Descartar tamaños inválidos y avisar
    invalid = df['Length'].isna() | (df['Length'] <= 0)
    if invalid.any():
        print(f'Aviso: se descartan {invalid.sum()} inversiones NAHR con coordenadas inválidas o tamaño <= 0:')
        print(df.loc[invalid, ['inversion', 'chr', 'bp1_2', 'bp2_1', 'Length']].to_string(index=False))
    df = df[~invalid].copy()
    df['Length'] = df['Length'].astype(int)

    # Mediating: '-' pasa a 'NA'
    df['Mediating'] = df['Mediating'].replace({'-': 'NA'}).fillna('NA')

    # Nombres de columnas compatibles con el pipeline original (beg/end = segmento interno)
    table = df.rename(columns={'chr': '#ref', 'bp1_2': 'beg', 'bp2_1': 'end'})
    table = table[['inversion', '#ref', 'beg', 'end', 'Length', 'Type', 'Mediating']]
    return table


def filter_sd(values, n_sd=2):
    """Elimina valores fuera de ±n_sd desviaciones estándar de la media (opcional)."""
    values = np.asarray(values)
    if len(values) < 3:
        return values
    mean, sd = values.mean(), values.std(ddof=1)
    return values[(values >= mean - n_sd * sd) & (values <= mean + n_sd * sd)]


def length_distribution(table, sd_filter=False):
    """Distribución global de tamaños (todos los cromosomas, incluido chrY si hubiera)."""
    lengths = table['Length'].to_numpy()
    if sd_filter:
        n0 = len(lengths)
        lengths = filter_sd(lengths)
        print(f'Filtro ±2 SD: {n0 - len(lengths)} inversiones eliminadas')

    # Tamaños crudos, uno por línea (fácil de leer desde el simulador)
    pd.DataFrame({'Length': lengths}).to_csv('Inversion_Lengths.tsv', sep='\t', index=False)

    # Resumen estadístico
    summary = pd.DataFrame([{
        'N': len(lengths), 'Min': lengths.min(), 'Q1': np.percentile(lengths, 25),
        'Median': np.median(lengths), 'Mean': lengths.mean(),
        'Q3': np.percentile(lengths, 75), 'Max': lengths.max(),
        'SD': lengths.std(ddof=1) if len(lengths) > 1 else np.nan,
    }])
    summary.to_csv('Length_Summary.tsv', sep='\t', index=False)
    return lengths


def mediating_proportions(table):
    """Proporción de inversiones según el mecanismo mediador (SD, elementos móviles...)."""
    counts = table['Mediating'].value_counts()
    df = pd.DataFrame({'Mediating': counts.index, 'N': counts.values,
                       'Probability': counts.values / counts.sum()})
    df.to_csv('Mediating_Proportions.tsv', sep='\t', index=False)


def mut_bins(bins, table):
    """Cuenta inversiones por ventana genómica."""
    rows = []
    for chrom, start, end in bins:
        n = ((table['#ref'] == chrom) & (table['beg'] >= start) & (table['end'] <= end)).sum()
        rows.append([chrom, start, end, n])
    return pd.DataFrame(rows, columns=['window', 'beg', 'end', 'Inversions'])


def read_chrom_lengths(path):
    """
    Lee el fichero de longitudes de cromosomas: nombre en la 1a columna y longitud en la 2a
    (separado por tabuladores o espacios; admite .fai u otras columnas extra; cabecera opcional).
    """
    df = pd.read_csv(path, sep=r'\s+', engine='python', header=None, usecols=[0, 1], names=['chr', 'length'])
    df['length'] = pd.to_numeric(df['length'], errors='coerce')
    df = df.dropna(subset=['length'])  # descarta una posible cabecera
    return dict(zip(df['chr'], df['length'].astype(int)))


def make_genomic_bins(chr_length, bin_size, chromosomes):
    """Ventanas consecutivas de bin_size pb por cromosoma (la última se recorta a la longitud)."""
    bins = []
    for chrom in chromosomes:
        if chrom not in chr_length:
            print(f'Aviso: {chrom} no está en el fichero de longitudes, se omite')
            continue
        for start in range(0, chr_length[chrom], bin_size):
            bins.append((chrom, start, min(start + bin_size, chr_length[chrom])))
    return bins


def genome_wide_distribution(chromosome_length, bin_size, table):
    """Distribución genómica de las inversiones por ventanas. Excluye chrY."""
    chr_length = read_chrom_lengths(chromosome_length)
    bins = make_genomic_bins(chr_length, bin_size, CHROMOSOMES)
    res = mut_bins(bins, table)

    total = res['Inversions'].sum()
    if total != 0:
        res['Inversions'] = res['Inversions'] / total
    res.to_csv('Genome_Wide_Distribution.tsv', sep='\t', index=False)


def main(file_path, chromosome_length, bin_size, seed, sd_filter):
    print(f'Fichero de inversiones: {file_path}')

    set_seed(seed)
    df = read_inversion_table(file_path)
    table = process_inversion_table(df)
    print(f'Inversiones NAHR válidas: {len(table)}')

    length_distribution(table, sd_filter)
    mediating_proportions(table)

    if chromosome_length:
        genome_wide_distribution(chromosome_length, bin_size, table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distribución de tamaños de inversiones NAHR a partir de una tabla de breakpoints.')
    parser.add_argument('--file_path', type=str, required=True, help='Tabla de inversiones.')
    parser.add_argument('--chromosome_length', type=str, required=False, default=None,
                        help='Fichero de longitudes de cromosomas (opcional, para la distribución genómica).')
    parser.add_argument('--bin_size', type=int, default=1000000, help='Tamaño de ventana (default: 1000000).')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria (default: 42).')
    parser.add_argument('--sd_filter', action='store_true',
                        help='Aplica el filtro de ±2 SD sobre los tamaños (desactivado por defecto).')
    args = parser.parse_args()
    main(args.file_path, args.chromosome_length, args.bin_size, args.seed, args.sd_filter)
