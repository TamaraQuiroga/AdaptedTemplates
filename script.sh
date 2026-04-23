arrFiles=("1_templates.py" "2_counterfactuals.py" "3_train.py" "4.scores_classification.py" "5_bias_nuevo.ipynb" "utils.py")


find . -mindepth 1 ° - name "script.sh" -exec rm -rf {} +

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
