#!/bin/sh
# Downloads the three open-licence fonts (SIL Open Font License) used by the
# lettering of the traced map into scratch/redraw/fonts/ (git-ignored).
#   IBM Plex Sans Condensed  (OFL)  names in brackets / MAROC, ALGÉRIE
#   Asap Condensed           (OFL)  territory names, title
#   Questrial                (OFL)  cities, region titles, legend (looks like the Gill Sans of the scan)
#   Lato                     (OFL)  bold legend heading and years
set -e
DIR="$(dirname "$0")/../scratch/redraw/fonts"
mkdir -p "$DIR"
B=https://github.com/google/fonts/raw/main/ofl
for f in ibmplexsanscondensed/IBMPlexSansCondensed-Regular.ttf \
         ibmplexsanscondensed/IBMPlexSansCondensed-SemiBold.ttf \
         asapcondensed/AsapCondensed-Bold.ttf \
         questrial/Questrial-Regular.ttf \
         lato/Lato-Bold.ttf; do
  curl -sL --max-time 60 -o "$DIR/$(basename "$f")" "$B/$f"
  file "$DIR/$(basename "$f")" | grep -q "TrueType" || { echo "download failed: $f" >&2; exit 1; }
done
echo "fonts in $DIR"
