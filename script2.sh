#!/bin/bash
set -e

REPO="$HOME/IMPORTANTE_SUBIR"
SOURCE="$HOME/template_domain"

arrFiles=("data")

# Ir al repositorio
cd "$REPO"

# Copiar archivos/carpetas
for file in "${arrFiles[@]}"; do
    echo "Copiando $file..."

    # borrar versión anterior para evitar data/data
    rm -rf "$REPO/$file"

    # copiar nueva versión
    cp -r "$SOURCE/$file" "$REPO/$file"
done

# Revisar cambios en el repositorio
if [ -n "$(git status --porcelain)" ]; then
    echo "Cambios detectados. Procediendo..."

    git add --all
    git commit -m "Update data files"
    git push
else
    echo "No hay cambios para subir"
fi