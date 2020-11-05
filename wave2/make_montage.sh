#!/bin/bash
montage -tile 2x2 -geometry 1800x1500+0+0 heatmap_ICU_r0_ifr.png heatmap_Tp_r0_ifr.png heatmap_Q_r0_ifr.png heatmap_Reff_r0_ifr.png montage.png
echo "DONE"
