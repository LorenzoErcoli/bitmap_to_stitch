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

Lo step `analyze_stats.py` ora copre anche la **FASE 7** con un riconoscimento euristico Run/Satin/Tatami per ogni layer (densità, lunghezza media e direzioni). Insieme otterrai: JSON IR con punti/color-change, statistiche tecniche (lunghezze, densità per layer, istogrammi angolari, guess tipo stitch), preview PNG e una Recipe Card JSON pronta per l'archiviazione delle lavorazioni.

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
- se passi `--description` aggiunge una nota testuale sfruttabile dal planner/AI
- crea `manifest.json` per il singolo item
- aggiorna `embroidery_library/library_index.json` così il Planner/AI può catalogare tutto via CLI.

Per importare **tutti** i `.dst` presenti in `embroidery_dst_lab/input/` con un solo comando usa lo script batch:

```
python embroidery_dst_lab/src/batch_build_library.py \
  --description-template "Auto import {name}"
```

Per ogni file genera automaticamente lo `slug` (item-id), controlla se la scheda esiste già e in caso contrario invoca la pipeline della Fase 8. Puoi usare `--rebuild-existing` per rigenerare card esistenti e `--skip-preview` se lavori in ambienti headless.

### FASE 9 — Planner/AI helper (ricerca rapida)

Una volta popolata la library puoi interrogare le schede con un singolo comando:

```
python embroidery_dst_lab/src/planner_query.py \
  --query "tatami canvas" \
  --require-class tatami \
  --max-results 3
```

Il tool legge `embroidery_library/library_index.json`, filtra per classificazione (`--require-class` ripetibile) e fa keyword search su label+descrizione restituendo i match più pertinenti con percorsi a recipe/preview. Usa `--json` se vuoi l'output in formato machine-friendly.
