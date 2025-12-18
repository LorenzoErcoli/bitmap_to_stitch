/**
 * Illustrator script
 * ------------------
 * - Usa il documento attualmente aperto.
 * - Per ogni path con stroke attivo crea un layer dedicato (nome = colore
 *   della linea con indice progressivo) senza alterare l'ordine di disegno.
 * - Salva automaticamente una copia AI8 accanto al file originale con il
 *   suffisso "_AI8.ai" senza aprire finestre di dialogo.
 *
 * ATTENZIONE: Il documento corrente viene modificato (i layer originali
 * saranno riorganizzati). Lavora su una copia se vuoi mantenere la struttura
 * precedente.
 */
function main() {
    if (app.documents.length === 0) {
        alert("Nessun documento aperto.");
        return;
    }

    var doc = app.activeDocument;
    if (!doc.saved) {
        alert("Salva il documento almeno una volta prima di lanciare lo script.");
        return;
    }

    var paths = collectStrokedPaths(doc);
    if (paths.length === 0) {
        alert("Nessuna path con stroke trovata.");
        return;
    }

    var colorMap = {};
    var processedLayers = [];
    var layerIndexCounters = {};
    for (var i = 0; i < paths.length; i++) {
        var info = paths[i];
        var item = info.item;
        try {
            if (item.locked) item.locked = false;
            if (item.hidden) item.hidden = false;
            var colorName = describeColor(item.strokeColor);
            var bucket = colorMap[colorName];
            if (!bucket) {
                var layerName = nextLayerName(colorName, layerIndexCounters);
                var newLayer = doc.layers.add();
                newLayer.name = layerName;
                bucket = {
                    layer: newLayer,
                    order: info.order
                };
                colorMap[colorName] = bucket;
                processedLayers.push(bucket);
            }
            item.move(bucket.layer, ElementPlacement.PLACEATEND);
        } catch (err) {
            // ignora l'oggetto problematico e continua
        }
    }

    if (processedLayers.length === 0) {
        alert("Nessuna path è stata spostata. Operazione annullata.");
        return;
    }

    reorderLayersByOrder(processedLayers, doc);

    var destFile = buildAi8Path(doc.fullName);
    var prevInteraction = app.userInteractionLevel;
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
    try {
        var saveOpts = new IllustratorSaveOptions();
        saveOpts.compatibility = Compatibility.ILLUSTRATOR8;
        saveOpts.flattenOutput = OutputFlattening.PRESERVEAPPEARANCE;
        saveOpts.pdfCompatible = false;
        saveOpts.embedICCProfile = false;
        doc.saveAs(destFile, saveOpts);
        alert("File AI8 creato: " + destFile.fsName);
    } catch (errSave) {
        alert("Errore salvataggio AI8: " + errSave);
    } finally {
        app.userInteractionLevel = prevInteraction;
    }
}

function collectStrokedPaths(doc) {
    var result = [];
    var items = doc.pathItems;
    for (var i = 0; i < items.length; i++) {
        try {
            var it = items[i];
            if (!it.stroked || !it.strokeColor) {
                continue;
            }
            if (it.strokeColor.typename === "NoColor" || it.guides || it.clipping) {
                continue;
            }
            var orderValue = 0;
            try {
                orderValue = it.zOrderPosition;
            } catch (orderErr) {
                orderValue = i;
            }
            result.push({
                item: it,
                order: orderValue
            });
        } catch (err) {
            // skip problematic item
        }
    }
    result.sort(function (a, b) {
        return a.order - b.order;
    });
    return result;
}

function reorderLayersByOrder(entries, doc) {
    entries.sort(function (a, b) {
        return a.order - b.order;
    });
    var anchor = doc.layers[doc.layers.length - 1];
    for (var i = 0; i < entries.length; i++) {
        var layer = entries[i].layer;
        layer.move(anchor, ElementPlacement.PLACEAFTER);
        anchor = layer;
    }
}

function buildAi8Path(originalFile) {
    var baseName = originalFile.displayName.replace(/\.[^\.]+$/, "");
    return new File(originalFile.path + "/" + baseName + "_AI8.ai");
}

function describeColor(color) {
    if (!color) {
        return "NoColor";
    }
    switch (color.typename) {
        case "RGBColor":
            return "RGB-" + toHex(Math.round(color.red)) + toHex(Math.round(color.green)) + toHex(Math.round(color.blue));
        case "CMYKColor":
            return "CMYK-" + Math.round(color.cyan) + "-" + Math.round(color.magenta) + "-" + Math.round(color.yellow) + "-" + Math.round(color.black);
        case "GrayColor":
            return "Gray-" + Math.round(color.gray);
        case "SpotColor":
            return "Spot-" + (color.spot ? color.spot.name : "Spot");
        case "PatternColor":
            return "Pattern-" + (color.pattern ? color.pattern.name : "Pattern");
        case "GradientColor":
            return "Gradient-" + (color.gradient ? color.gradient.name : "Gradient");
        default:
            return color.typename;
    }
}

function toHex(value) {
    var v = Math.max(0, Math.min(255, value));
    var hex = v.toString(16);
    return hex.length === 1 ? "0" + hex : hex;
}

function nextLayerName(baseName, counters) {
    var key = baseName || "Layer";
    if (!counters[key]) {
        counters[key] = 1;
        return key;
    }
    counters[key] += 1;
    return key + " #" + counters[key];
}

main();
