General workflow (summarised in atlas_integration_overview.ipynb for obtaining fibroblasts)
1. Integrate all skin cells (loaded via 0_load_all_data.py) after running cellbender (0_run_cellgender_gse.sh). Scrublet on dataset to exclude doublets.
2. Select fibroblasts
3. Regenerate scVI embeddings for healthy fibroblasts only
4. Map disease fibroblasts to healthy reference using scPoli

Additional
- NicheCompass to identify superficial perivascular niche for F3 (Fig. 7)
- Mapping and annotation of wound data (Fig. 5)
- Load velocity data for datasets where this is available (Fig. 5)