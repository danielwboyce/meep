import matplotlib.pyplot as plt
import numpy as np


resolutions = [15, 20, 25, 30, 40, 45]

smoothed_relative_errors = [0.04164297127159183, 0.07000770931926027, 0.05913375399411488,
                            0.037019266031992515, 0.01309392798269891, 0.00465032393533884]

unsmoothed_relative_errors = [0.041642971271583534, 0.07000770931925482, 0.05913375399412627,
                              0.03701926613241959, 0.01309392798270708, 0.0046503239353362395]

plt.figure(dpi=150)
plt.loglog(resolutions, smoothed_relative_errors, 'bo-', label='With Pixel Smoothing')
plt.loglog(resolutions, unsmoothed_relative_errors, 'ro-', label='Without Pixel Smoothing')
plt.grid(True,which="both",ls="-")
plt.xlabel('resolution (pixels/μm)')
plt.ylabel('relative error in MEEP calculation')
plt.legend(loc='lower left')
plt.title('Relative Error in Mie Absorption Efficiency, \nComparing MEEP and PyMieScatt')
plt.tight_layout()
#plt.show()
plt.savefig("mie_absorption_error.png")
