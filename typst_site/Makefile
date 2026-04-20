# Find all .typ files in content/ that don't start with an underscore in their path
TYP_FILES := $(shell find content -name '*.typ' -not -path '*/_*')

# Generate corresponding HTML file paths in _site/
HTML_FILES := $(patsubst content/%.typ,_site/%.html,$(TYP_FILES))

# The main target 'html' depends on a fresh site dir, generated publications, resume PDF, HTML files, and assets
html: prepare-site publications cv-pdf $(HTML_FILES) assets

# Preview settings
PORT ?= 8000

# Pattern rule to compile .typ files to .html files
# $< is the first prerequisite (the .typ file)
# $@ is the target (the .html file)
# $(@D) is the directory part of the target
_site/%.html: content/%.typ
	@mkdir -p $(@D)
	typst compile --root .. --features html --format html $< $@

prepare-site:
	@rm -rf _site
	@mkdir -p _site

assets:
	@mkdir -p _site/assets
	@cp -r assets/* _site/assets/

publications:
	python3 scripts/generate_publications.py

cv-pdf:
	typst compile --root .. ../typst_resume/main.typ assets/EricBoittier_CV.pdf

# A clean rule to remove generated files
clean:
	rm -rf _site

preview: html
	@echo "Serving _site at http://localhost:$(PORT)"
	@cd _site && python -m http.server $(PORT)

.PHONY: html clean assets publications cv-pdf prepare-site preview
