# bitmap_to_stitch

Utility per convertire bitmap B/N in un'unica path SVG pensata per macchine da ricamo.

## Utilizzo CLI

```
python bitmap_to_stitch.py input.png output.svg \
  --style carpet --grid-cell-size 4 \
  --ordering scanline --scanline-band-height 4 \
  --min-dist 2.5 --path-chunk-size 5000 \
  --max-points 30000
```

Passando `--style degrade` si ottiene l'effetto casuale/degradé, mentre le altre opzioni controllano sampling, distanza minima e chunking della path.

## Interfaccia Web (PyScript)

È inclusa una piccola app web in `web/index.html` che gira interamente nel browser grazie a PyScript:

1. Apri `web/index.html` direttamente nel browser, oppure punta GitHub Pages alla cartella `web/` per renderla accessibile online.
2. Carica la bitmap, scegli lo stile (carpet o degradé) e premi **Genera SVG**.
3. Scarica il risultato tramite il link fornito.

La logica Python riusa le stesse funzioni dello script principale, quindi puoi ottenere rapidamente un risultato anche da dispositivi senza Python installato.

## Laboratorio DST (CLI puro)

Per l'analisi dei file `.dst` trovi una pipeline autonoma nella cartella `embroidery_dst_lab/` pensata come laboratorio da terminale:

1. Installa le dipendenze consigliate: `pip install pyembroidery numpy shapely matplotlib`.
2. Posiziona i file DST in `embroidery_dst_lab/input/`.
3. Fasi principali (tutte CLI):
   - `python embroidery_dst_lab/src/read_dst.py input/test.dst`
   - `python embroidery_dst_lab/src/dst_to_ir.py input/test.dst output/test_stitch_ir.json`
   - `python embroidery_dst_lab/src/analyze_stats.py --ir output/test_stitch_ir.json`
   - `python embroidery_dst_lab/src/ir_to_recipe.py --stats output/test_stats.json`

Lo step `analyze_stats.py` ora copre anche la **FASE 7** con un riconoscimento euristico multi-classe per ogni layer (run, travel, satin_light, satin_dense, tatami, detail) basato su densità, lunghezze, deviazione standard e direzioni. Insieme otterrai: JSON IR con punti/color-change, statistiche tecniche (lunghezze, densità per layer, istogrammi angolari, guess tipo stitch), preview PNG e una Recipe Card JSON pronta per l'archiviazione delle lavorazioni.

### FASE 8 — Esportazione nella Library per l'AI Planner

Pacchettizza automaticamente tutti gli artifact (DST, IR, stats, preview, recipe JSON/YAML, manifest) dentro `embroidery_library/items/<id>/` eseguendo:

```
python embroidery_dst_lab/src/build_library_item.py \
  --dst embroidery_dst_lab/input/BASE\ PUNTO\ CANVAS\ 2\ PASS.dst \
  --item-id base_punto_canvas_2_pass \
  --description "Base canvas stitch fill"
```

Lo script:

- copia il DST nella cartella item
- rigenera IR/stats/preview/recipe
- salva `recipe.json` + `recipe.yaml` con summary dei layer e copia il nome file DST come label
- se presenti, aggiunge `artwork.*` (grafica origine) e `stitched_photo.*` (foto ricamo) passando `--artwork/--stitched-photo` o usando il batch con auto-attach
- accetta metadati opzionali (`--subject`, `--tag` ripetibile, `--fabric`, `--thread`, `--stabilizer`, `--quality-score`, `--notes`) salvandoli in manifest/index/recipe
- se non passi `--description` costruisce in automatico una descrizione testuale dalle proprietà dei layer (classe, densità, lunghezza media)
- se passi `--description` aggiunge una nota testuale sfruttabile dal planner/AI
- crea `manifest.json` per il singolo item
- aggiorna `embroidery_library/library_index.json` così il Planner/AI può catalogare tutto via CLI.

Per importare **tutti** i `.dst` presenti in `embroidery_dst_lab/input/` con un solo comando usa lo script batch:

```
python embroidery_dst_lab/src/batch_build_library.py \
  --description-template "Auto import {name}"
```

Per ogni file genera automaticamente lo `slug` (item-id), controlla se la scheda esiste già e in caso contrario invoca la pipeline della Fase 8. Puoi usare `--rebuild-existing` per rigenerare card esistenti e `--skip-preview` se lavori in ambienti headless.
Puoi allegare automaticamente artwork e foto (se trovati accanto al DST) con `--auto-artwork --auto-stitched-photo` e personalizzare i suffissi cercati con `--artwork-suffixes _artwork` / `--stitched-photo-suffixes _stitched,_photo,_ricamo`. Gli stessi metadati del comando singolo sono disponibili in batch (applicati a tutti gli item).

### FASE 9 — Planner/AI helper (ricerca rapida)

Una volta popolata la library puoi interrogare le schede con un singolo comando:

```
python embroidery_dst_lab/src/planner_query.py \
  --query "tatami canvas" \
  --require-class tatami \
  --max-results 3
```

Il tool legge `embroidery_library/library_index.json`, filtra per classificazione (`--require-class` ripetibile), per range di stitch (`--min-stitches/--max-stitches`) e per densità media (`--min-density/--max-density`), poi fa keyword search su label+descrizione restituendo i match più pertinenti con percorsi a recipe/preview. Usa `--json` se vuoi l'output in formato machine-friendly.
Ora puoi anche filtrare per `--subject`, `--tag`, `--fabric`, `--stabilizer`, e per qualità (`--min-quality/--max-quality`), ottenendo in output eventuali percorsi a `artwork`/`stitched_photo` se presenti.

### Embedding visivi (offline semplice)

Per creare embedding dalle immagini `artwork`/`stitched_photo` e abilitarne il retrieval visivo:

```
python embroidery_dst_lab/src/build_image_embeddings.py --library-root embroidery_library
```

Genera `embroidery_library/image_embeddings.json` con vettori normalizzati (modello `simple_v1` basato su resize+istogrammi/downsample). È offline e senza dipendenze extra. Se in futuro installi un modello visivo (es. CLIP), puoi estendere lo script per usare embedding più potenti mantenendo lo stesso file di output.

Fai nearest-neighbor sui vettori con:

```
python embroidery_dst_lab/src/image_retrieval.py \
  --library-root embroidery_library \
  --query-item damsco_floreale_punto_mattezzo_con_base_per_appiattimento_filo_effetto_vintage \
  --prefer-asset artwork \
  --top-k 5
```

oppure con un’immagine esterna:

```
python embroidery_dst_lab/src/image_retrieval.py \
  --library-root embroidery_library \
  --query-image /path/to/nuova_grafica.png \
  --top-k 5
```

### Embedding visivi CLIP + retrieval ibrido (testo + immagine)

Installa le dipendenze CLIP (CPU): `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && pip install open_clip_torch pillow`

Genera embedding CLIP (default ViT-B-32 openai) su artwork/foto:

```
python embroidery_dst_lab/src/build_image_embeddings_clip.py --library-root embroidery_library --model ViT-B-32 --pretrained openai
```

Salva `embroidery_library/image_embeddings_clip.json`.

Retrieval ibrido (testo + immagine) usando CLIP:

```
python embroidery_dst_lab/src/hybrid_retrieval.py \
  --library-root embroidery_library \
  --embeddings-file embroidery_library/image_embeddings_clip.json \
  --model clip:ViT-B-32:openai \
  --query "floreale velluto tatami" \
  --query-image embroidery_dst_lab/queries/nuova_grafica.png \
  --top-k 5
```

Se passi solo `--query` senza `--query-image`, fa ranking testuale su label/description/metadati. Con `--query-image` reranka i migliori per testo usando la similarità CLIP delle immagini.

### Suggeritore ricetta (testo + immagine) con recipe finale

Restituisce i match migliori e carica la recipe del top-1, arricchita con `suggested_from`:

```
python embroidery_dst_lab/src/suggest_recipe.py \
  --library-root embroidery_library \
  --embeddings-file embroidery_library/image_embeddings_clip.json \
  --model clip:ViT-B-32:openai \
  --query "floreale velluto tatami" \
  --query-image embroidery_dst_lab/queries/nuova_grafica.png \
  --top-k 3 \
  --save-suggestion embroidery_dst_lab/output/suggested_recipe.json
```

Senza `--query-image` usa solo il ranking testuale; con `--query-image` combina punteggio testo + similarità immagine.

### Merge ricette assistito da OpenAI (sceglie layer dai top-k)

Installare le dipendenze: `pip install openai`

Poi:

```
OPENAI_API_KEY=... python embroidery_dst_lab/src/ai_merge_recipe.py \
  --library-root embroidery_library \
  --embeddings-file embroidery_library/image_embeddings_clip.json \
  --model clip:ViT-B-32:openai \
  --query "floreale velluto tatami" \
  --query-image embroidery_dst_lab/queries/nuova_grafica.png \
  --top-k 3 \
  --openai-model gpt-4o-mini \
  --save embroidery_dst_lab/output/ai_suggested_recipe.json
```

Il flow: retrieval (testo + opzionale immagine) → invio a OpenAI dei candidati con i loro layer → il modello restituisce un piano di merge (quali layer prendere da quali item) → lo script costruisce una recipe fusa con i layer indicati e la salva.
Se vuoi includere anche item senza immagini quando usi `--query-image`, aggiungi `--allow-text-only` (saranno ordinati solo per punteggio testo).

### (Prototype) Genera DST da una recipe + bitmap mask

Naive hatch fill basato su densità/avg stitch del layer:

```
python embroidery_dst_lab/src/recipe_to_dst.py \
  --recipe embroidery_dst_lab/output/ai_suggested_recipe.json \
  --mask embroidery_dst_lab/queries/nuova_grafica_mask.png \
  --pixel-size-mm 0.2 \
  --output-dst embroidery_dst_lab/output/suggested.dst \
  --output-ir embroidery_dst_lab/output/suggested_ir.json
```

Richiede `pyembroidery` e `pillow`. Usa il layer stack della recipe e riempie la maschera con hatching lineare (prototipo per testare la pipeline end-to-end).

### (Prototype) Segmenta immagine in regioni colore → mask + JSON

```
python embroidery_dst_lab/src/segment_image.py \
  --image embroidery_dst_lab/queries/nuova_grafica.png \
  --k 5 \
  --min-area-pct 0.5 \
  --output-dir embroidery_dst_lab/output/segments
```

Produce mask binarie per ciascuna regione colore (> soglia area) in `output/segments/masks/*.png` e un `segments.json` con bbox, area e colore medio, utile per collegare soggetti/stop ago.

### Retrieval per regione (usa le mask e gli embedding)

```
python embroidery_dst_lab/src/region_retrieval.py \
  --segments embroidery_dst_lab/output/segments/segments.json \
  --image embroidery_dst_lab/queries/nuova_grafica.png \
  --embeddings-file embroidery_library/image_embeddings_clip.json \
  --model clip:ViT-B-32:openai \
  --top-k 3 \
  --output embroidery_dst_lab/output/region_retrieval.json
```

Per ogni mask estrae il crop/alpha e fa nearest-neighbor sugli embedding (simple o CLIP) salvando i top match per regione.

### Suggerisci mini-recipe per regione con OpenAI

Richiede `openai` e `OPENAI_API_KEY` impostata. Usa i top match per regione (da `region_retrieval.json`) e chiede al modello quali layer di quale item riusare.

```
OPENAI_API_KEY=... python embroidery_dst_lab/src/region_suggest_recipe.py \
  --region-results embroidery_dst_lab/output/region_retrieval.json \
  --library-root embroidery_library \
  --top-k 3 \
  --openai-model gpt-4o-mini \
  --output embroidery_dst_lab/output/region_suggestions.json
```

Per ogni regione ottieni `selection` (item_id + layer_indices) e i layer copiati nella mini-recipe della regione.

### Genera DST multi-regione dalle mini-recipe

Usa le suggestion per regione (con mask) e genera un DST unico (tatami/satin prototipo):

```
python embroidery_dst_lab/src/region_to_dst.py \
  --region-suggestions embroidery_dst_lab/output/region_suggestions.json \
  --pixel-size-mm 0.2 \
  --output-dst embroidery_dst_lab/output/regions_suggested.dst \
  --output-ir embroidery_dst_lab/output/regions_suggested_ir.json \
  --output-svg embroidery_dst_lab/output/regions_suggested.svg
```

Richiede che `region_suggest_recipe.py` abbia incluso `mask` per ogni regione (lo fa di default). Fill tatami/satin ancora semplice; serve travel planner/colonne satin più evolute per uso reale.
