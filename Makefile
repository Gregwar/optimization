
all: pages zips

# Building pages
.PHONY: pages
pages: 
	for page in pages/*.md; do \
		page_html=$$(basename $$page .md).html ; \
		php $$page > /tmp/md_out.md ; \
		pandoc -H tpl/headers.html -B tpl/body_before.html -A tpl/body_after.html \
			  --highlight-style=zenburn -s --mathjax -o $$page_html /tmp/md_out.md ; \
	done
	
# Building zips
.PHONY: zips
zips:
	cd files ; \
	for dir in *; do \
		if [ -d $$dir ]; then \
			rm $$dir.zip ; \
			rm -rf $$dir/__pycache__; \
			zip -r $$dir.zip $$dir ; \
		fi ; \
	done
