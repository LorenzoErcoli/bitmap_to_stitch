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
