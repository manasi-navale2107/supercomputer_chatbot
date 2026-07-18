Place exactly these five CSV files in this folder:

1. `green500_systems.csv`
2. `top500_systems.csv`
3. `system_details.csv`
4. `top500_stat.csv`
5. `country_stats.csv`

The bootstrap command normalizes their column names, imports each file into a
same-named MySQL table, and creates Qdrant embeddings for semantic retrieval.

