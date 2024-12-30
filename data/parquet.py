
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.parquet.encryption as pe
from data.env import PK, PK2

from typing import List
from pyarrow.tests.parquet.encryption import InMemoryKmsClient

KMS_CC = pe.KmsConnectionConfig(
            custom_kms_conf={"footer": PK, "columns": PK})

KMS_CC2 = pe.KmsConnectionConfig(
            custom_kms_conf={"footer": PK2, "columns": PK2})


def read(in_file: str,
         usecols: List[str]=[], index_cols: List[str]=[]) -> pd.DataFrame:

    try:
        cripto_factory = pe.CryptoFactory(InMemoryKmsClient)
        decrypt_prop = cripto_factory.file_decryption_properties(KMS_CC)
        parquet_file_data = pq.ParquetFile(
            in_file, decryption_properties=decrypt_prop)
    except ValueError:
        cripto_factory = pe.CryptoFactory(InMemoryKmsClient)
        decrypt_prop = cripto_factory.file_decryption_properties(KMS_CC2)
        parquet_file_data = pq.ParquetFile(
            in_file, decryption_properties=decrypt_prop)

    df = parquet_file_data.read().to_pandas()
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    usecols = [_ for _ in set(usecols + index_cols)]
    df = df[usecols] if len(usecols) else df
    df = df.set_index(index_cols) if len(index_cols) else df

    return df


def write(out_file: str, df: pd.DataFrame) -> None:

    encryption_config = pe.EncryptionConfiguration(
            footer_key="footer",
            column_keys={"columns": df.columns.tolist()},
            encryption_algorithm="AES_GCM_V1",
            data_key_length_bits=128)

    cripto_factory = pe.CryptoFactory(InMemoryKmsClient)
    encryption_properties = cripto_factory.file_encryption_properties(
        KMS_CC, encryption_config)

    table = pa.Table.from_pandas(df)
    writer = pq.ParquetWriter(
            out_file,
            table.schema,
            encryption_properties=encryption_properties, 
            compression='SNAPPY'
        )
    writer.write_table(table)


if __name__ == '__main__':

    _, in_file = sys.argv
    out_file = in_file.replace('.csv', '.parquet')
    df = pd.read_csv(in_file, header=0)
    write(out_file, df)
