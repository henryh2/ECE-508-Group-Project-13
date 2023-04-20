#!/bin/bash

set -euo pipefail

echo cit-Patents     && wget -O cit-Patents.tar.zst    https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/cit-Patents.tar.zst    && tar --use-compress-program=unzstd -xvf cit-Patents.tar.zst
echo datagen-7_5-fb  && wget -O datagen-7_5-fb.tar.zst https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_5-fb.tar.zst && tar --use-compress-program=unzstd -xvf datagen-7_5-fb.tar.zst
echo datagen-7_6-fb  && wget -O datagen-7_6-fb.tar.zst https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_6-fb.tar.zst && tar --use-compress-program=unzstd -xvf datagen-7_6-fb.tar.zst
echo datagen-7_7-zf  && wget -O datagen-7_7-zf.tar.zst https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_7-zf.tar.zst && tar --use-compress-program=unzstd -xvf datagen-7_7-zf.tar.zst
echo datagen-7_8-zf  && wget -O datagen-7_8-zf.tar.zst https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_8-zf.tar.zst && tar --use-compress-program=unzstd -xvf datagen-7_8-zf.tar.zst
echo datagen-7_9-fb  && wget -O datagen-7_9-fb.tar.zst https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/datagen-7_9-fb.tar.zst && tar --use-compress-program=unzstd -xvf datagen-7_9-fb.tar.zst
echo dota-league     && wget -O dota-league.tar.zst    https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/dota-league.tar.zst    && tar --use-compress-program=unzstd -xvf dota-league.tar.zst
echo graph500-22     && wget -O graph500-22.tar.zst    https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/graph500-22.tar.zst    && tar --use-compress-program=unzstd -xvf graph500-22.tar.zst
echo kgs             && wget -O kgs.tar.zst            https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/kgs.tar.zst            && tar --use-compress-program=unzstd -xvf kgs.tar.zst
echo wiki-Talk       && wget -O wiki-Talk.tar.zst      https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/wiki-Talk.tar.zst      && tar --use-compress-program=unzstd -xvf wiki-Talk.tar.zst
