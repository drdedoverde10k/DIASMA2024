# ------------------------------------------------------------
# 0. Librerías y SparkSession
import re, time
from pyspark.sql import SparkSession, functions as F, types as T

spark = (
    SparkSession.builder
    .appName("FASTA_Kmer_Matching")
    .master("local[*]")
    .config("spark.driver.memory", "4g")   # ajusta según tu RAM
    .getOrCreate()
)

# ------------------------------------------------------------
# 1. Rutas de archivos
fasta_reads = "/ABSOLUTA/RUTA/A/tus_secuencias/*.fasta"
fasta_genome = "/ABSOLUTA/RUTA/A/Genoma_referencia.fasta"

# ------------------------------------------------------------
# 2. Función de parseo FASTA (RD → DF id,seq)
def parse_fasta(rdd):
    def parser(it):
        h, buf = None, []
        for line in it:
            l = line.strip()
            if l.startswith(">"):
                if h:
                    yield (h, "".join(buf))
                h, buf = l[1:], []
            else:
                buf.append(l)
        if h:
            yield (h, "".join(buf))
    return (rdd.mapPartitions(parser)
               .toDF(["id", "seq"]))

# -- leer reads
lines_reads = spark.read.text(fasta_reads).rdd
df_reads = parse_fasta(lines_reads)

# -- leer genoma y difundir
with open(fasta_genome) as f:
    genome_ref = "".join(l.strip() for l in f if not l.startswith(">"))
bc_gen = spark.sparkContext.broadcast(genome_ref)

# ------------------------------------------------------------
# 3. UDF que cuenta calces para un porcentaje dado
def make_udf(pct: float):
    """
    pct: entre 0.0 y 1.0   (ej. 0.25 = 25 %)
    devuelve un entero (# de calces)
    """
    @F.udf(T.IntegerType())
    def n_hits(seq: str):
        g = bc_gen.value
        k = max(1, int(round(len(seq) * pct)))
        total = 0
        # recorrer k-mers solapados
        for i in range(0, len(seq) - k + 1):
            frag = seq[i : i + k]
            total += len(list(re.finditer(re.escape(frag), g)))
        return total
    return n_hits

# ------------------------------------------------------------
# 4. Porcentajes que quieres probar  (0 %…100 %)
percentages = [0.1, 0.25, 0.5, 0.75, 1.0]   # cambia o genera con numpy

results = {}
for pct in percentages:
    col_name = f"matches_{int(pct*100):02d}"
    udf_hits = make_udf(pct)

    t0 = time.time()
    df_reads = df_reads.withColumn(col_name, udf_hits("seq"))
    spark.time()  # registra en el log de Spark

    elapsed = time.time() - t0
    results[pct] = elapsed
    print(f"✓ porcentaje {pct:4.0%}  (k variable) listo  |  tiempo: {elapsed:6.1f} s")

# ------------------------------------------------------------
# 5. Muestra las primeras lecturas con sus conteos
cols = ["id", "len"] + [f"matches_{int(p*100):02d}" for p in percentages]
(df_reads
 .withColumn("len", F.length("seq"))
 .select(cols)
 .show(10, truncate=False))

# ------------------------------------------------------------
# 6. Tabla de tiempos por porcentaje
print("\n--- Tiempo por pasada (s) ---")
for pct, t in results.items():
    print(f"{pct:4.0%} : {t:6.1f} s")


# Ruta HDFS, S3 o local (según tu clúster)
output_path = "hdfs:///user/fespinoza/calces_fasta.parquet"

(df_reads
 .withColumn("len", F.length("seq"))         # si aún no la tienes
 .write
 .mode("overwrite")                          # o "append"
 .parquet(output_path))

spark.stop()  # si cierras el script