# Stato attuale del progetto

Documento di fotografia tecnica del progetto prima delle prossime migliorie.

## Scopo

`bitmap_to_stitch` e' uno strumento per trasformare immagini bitmap in tracciati
SVG utilizzabili come base per lavorazioni di ricamo.

Il progetto oggi contiene due linee di lavoro:

- conversione bitmap -> punti -> SVG, usabile da CLI, browser e app locale;
- laboratorio DST/recipe/library per analizzare file ricamo esistenti e usarli
  come base per un planner assistito da AI.

## Conversione bitmap -> SVG

La pipeline principale prende un'immagine, seleziona i pixel utili, li converte
in punti, ordina questi punti e genera percorsi SVG.

Funzioni principali:

- selezione pixel tramite soglia di luminosita';
- selezione aggiuntiva tramite colori campionati;
- quantizzazione colore per separare l'immagine in piu' tinte;
- output SVG multicolore, monocolore e separato per singolo colore;
- ordinamento punti `scanline` o `nearest`;
- stile `carpet` per distribuzione regolare;
- stile `degrade` con scarto e jitter casuale;
- limite massimo punti e densita' globale;
- distanza minima tra punti consecutivi espressa in mm;
- reinserimento dei punti filtrati quando rispettano la distanza minima;
- chunking delle path SVG per alleggerire file grandi.

File coinvolti:

- `bitmap_to_stitch.py`: CLI originale per conversione bitmap -> SVG.
- `web/app.py`: core della pipeline web, con supporto multicolore.
- `web/main.js`: UI browser, lettura opzioni, download output.
- `web/index.html`: interfaccia utente.

## App web e app locale

La web app puo' funzionare in due modi:

1. Modalita' browser/Pyodide: se aperta come pagina statica, carica Pyodide dal
   CDN e fa girare Python nel browser.
2. Modalita' locale: se aperta da `localhost` o `127.0.0.1`, usa il backend
   Python locale esposto da `local_server.py`.

`local_server.py` avvia un server su `127.0.0.1:8765`, serve i file della
cartella `web/` ed espone:

- `GET /health`: stato del server e percorso log;
- `POST /convert`: riceve immagine base64 + opzioni e restituisce SVG/summary.

La build Windows impacchetta `local_server.py` e `web/` tramite PyInstaller.

File coinvolti:

- `local_server.py`
- `run_local_app.bat`
- `build_windows.bat`
- `bitmap_to_stitch_local.spec`
- `.github/workflows/build-windows.yml`

## Pubblicazione

La cartella `docs/` contiene una copia pubblicabile della web app per GitHub
Pages.

La release Windows viene costruita dalla workflow GitHub Actions quando viene
pushato un tag `v*`. La workflow genera `BitmapToStitch-Windows.zip` e lo
pubblica nella release del tag.

## Laboratorio DST / recipe / library

La cartella `embroidery_dst_lab/` contiene una pipeline sperimentale per
analizzare file `.dst` esistenti e trasformarli in una libreria interrogabile.

Funzioni principali:

- lettura DST con `pyembroidery`;
- conversione DST -> Stitch IR JSON;
- analisi statistiche per layer;
- classificazione euristica dei layer;
- generazione preview PNG;
- generazione recipe JSON/YAML;
- pacchettizzazione in `embroidery_library/items/<id>/`;
- indicizzazione in `embroidery_library/library_index.json`;
- retrieval testuale;
- embedding visivi semplici offline;
- embedding CLIP opzionali;
- retrieval ibrido testo + immagine;
- suggerimento ricetta;
- merge ricette assistito da OpenAI;
- prototipi per segmentare immagini e generare DST da recipe + mask.

File principali:

- `embroidery_dst_lab/src/dst_to_ir.py`
- `embroidery_dst_lab/src/analyze_stats.py`
- `embroidery_dst_lab/src/ir_to_recipe.py`
- `embroidery_dst_lab/src/build_library_item.py`
- `embroidery_dst_lab/src/batch_build_library.py`
- `embroidery_dst_lab/src/planner_query.py`
- `embroidery_dst_lab/src/build_image_embeddings.py`
- `embroidery_dst_lab/src/build_image_embeddings_clip.py`
- `embroidery_dst_lab/src/hybrid_retrieval.py`
- `embroidery_dst_lab/src/suggest_recipe.py`
- `embroidery_dst_lab/src/ai_merge_recipe.py`
- `embroidery_dst_lab/src/segment_image.py`
- `embroidery_dst_lab/src/region_retrieval.py`
- `embroidery_dst_lab/src/region_suggest_recipe.py`
- `embroidery_dst_lab/src/recipe_to_dst.py`
- `embroidery_dst_lab/src/region_to_dst.py`

## Stato repository

Al momento della fotografia:

- branch locale: `ai-integrator`;
- remote: `https://github.com/LorenzoErcoli/bitmap_to_stitch.git`;
- working tree pulito prima della creazione di questo documento;
- ultimo pacchetto locale presente: `BitmapToStitch-Windows.zip`.

## Limiti e debito tecnico

Punti da sistemare prima di evolvere il progetto in modo robusto:

- `web/` e `docs/app/` duplicano gli stessi file applicativi;
- parte della logica e' duplicata tra `bitmap_to_stitch.py` e `web/app.py`;
- mancano test automatici;
- mancano file dipendenze standard come `requirements.txt` o `pyproject.toml`;
- alcuni testi mostrano problemi di encoding;
- la parte DST -> recipe -> DST e' ancora prototipale;
- non esiste ancora una separazione chiara tra core library, UI web e tool CLI;
- la modalita' Pyodide dipende dal CDN esterno;
- il README contiene molta documentazione storica e potrebbe essere separato in
  guide piu' piccole.

## Direzione consigliata per le migliorie

Prima di aggiungere nuove funzioni conviene stabilizzare la base:

1. creare un branch/fork di lavoro;
2. aggiungere `requirements.txt` o `pyproject.toml`;
3. estrarre la pipeline comune in un modulo unico riusabile da CLI, web e server;
4. automatizzare la sincronizzazione o eliminare la duplicazione tra `web/` e
   `docs/app/`;
5. aggiungere test minimi sulla conversione immagine -> SVG;
6. correggere encoding e testi UI;
7. decidere se il laboratorio DST deve restare prototipo o diventare prodotto.

