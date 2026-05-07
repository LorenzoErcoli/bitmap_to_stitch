# Release Bitmap To Stitch

## Build locale Windows

```powershell
.\build_windows.bat
Compress-Archive -Path dist\BitmapToStitch\* -DestinationPath BitmapToStitch-Windows.zip -Force
```

Lo zip generato contiene `BitmapToStitch.exe` e tutte le dipendenze.

## Release GitHub automatica

1. Commit e push delle modifiche.
2. Crea un tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

3. GitHub Actions costruisce `BitmapToStitch-Windows.zip`.
4. Lo zip viene pubblicato nella release del tag.

## GitHub Pages

Pubblica la cartella `docs/` da:

`Settings -> Pages -> Build and deployment -> Source: Deploy from a branch -> main / docs`

La pagina di download sara':

`https://lorenzoercoli.github.io/bitmap_to_stitch/`
