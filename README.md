# Segmentace buněk v BAL snímcích pomocí strojového učení

Tento repozitář obsahuje zdrojové kódy a datové soubory pro diplomovou práci.
Diplomová práce se zabývá segmentací buněk ve snímcích BAL. 

## Obsah

- Data: Složka se snímky bronchoalveolární laváže
- Results: Složka, která obsahuje podsložky s dílčími výsledky
    - Analysis:
    - Best_Models: Výsledky segmentace ML algortimů s optimálními hyperparametry
    - Clustering_Features: Výsledné CSV soubory po shlukování features
    - Generating_Features: Vygenerované CSV soubory features
    - Hyperparameters_Tuning: Výsledky RayTune - CSV soubory 
    - Other_ML_Algorithms: Výsledky segmentace ostatních ML algoritmů s defaultnímy parametry 
    - Resizing_Images: Snímky, které mají změněné rozlišení na $500 \times 500$ a jejich anotace
    - Selecting_Features: CSV soubory, které obsahují jen vybraný počet features
    - Workflow_Testing: Obsahuje podsložky s dílčími výsledky postupu segmentace. Podsloužky obsahují CSV soubory a obrázky metrik.
     
- Scripts: Složka, která obsahuje podsložky se skripty a 
    - Analysis: Obsahuje podsložky s Jupyter Notebooky, které obsahují vizualizaci k problematice dané podsložky
    - Best_Models: Obsahuje skripty, kde jsou ML algoritmy s nastavenými hyperparametry
    - Clustering_Features: Obsahuje skripty, které shlukují feature space
    - Filter_Bank: Obsahuje Jupyter Notebooky s nazvy, které odpovídají kapitolám v diplomové práci. Ukazuje aplikace filtrů
    - Generating_Features: Obsahuje skripty, které ze snímků vytvoří CSV soubory (feature space)
    - Hyperparameters_Tuning: Obsahuje skripty, kde se hledají hyperparametry SVM a RF
    - Other_ML_Algorithms: Obsahuje skripty, kde ostatní ML algoritmy s defaultními parametry 
    - Resizing_Images: Obsahuje skript, který mění rozlišení snímků
    - Selecting_Features: Obsahuje skripty, který vybírají features 
    - Workflow_Testing: Obsahuje podsložky se skripty, který testují kroky postupu segmentace a Jupyter Notebooky s vyzualizací dílčích výsledků.

### Poznámka ke složce Results
Složky [Clustering_Features, Generating_Features, Selecting_Features] neobsahují kompletní výsledky. Výsledky jsou zde ukázány pouze částečně nebo vůbec z důvodu velké paměťové náročnosti. Pro vygenerování výsledků je nutné spustit skripty v příslušných podsložkách. 

1. **feature_generator.py**
2. **feature_clusterer_CPU.py** nebo **feature_clusterer_GPU.py**
3. **feature_selection.py** a **feature_selection_based_on_experiment.py**

Spuštění těchto skriptů je časově i paměťově náročné (celkově hodiny a cca 100 GB).

## Použití

Pro práci s notebooky a skripty je potřeba mít nainstalovaný Jupyter Notebook a Python.

Jednotlivé knihovny naleznete v souboru requirements.txt, některé skripty využívají akceleraci na GPU pomocí technologie RAPIDS viz https://docs.rapids.ai/install.






