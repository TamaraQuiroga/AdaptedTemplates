arrFiles=("data")
for file in "${arrFiles[@]}"; do
	cp -r "$HOME/template_domain/$file" "$HOME/IMPORTANTE_SUBIR/$file"
done

if [ -n "$(git status --porcelain)" ]; then
	echo "Cambios detectados. Procediendo..."
	git add --all
	git commit -m "Auto commit $(date)"
	git push
else
	echo "No hay cambios para subir"
fi
