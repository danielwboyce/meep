import matplotlib.pyplot as plt


resolutions = [10, 15, 20, 25, 30, 35]

smoothed_relative_errors = [1.1429734443065664, 0.04164297127159183, 0.07000770931926027, 0.05913375399411488, 0.037019266031992515, 1.7848960719433735]

unsmoothed_relative_errors = [1.1429734443065573, 0.041642971271583534, 0.07000770931925482, 0.07000770931925482, 0.03701926613241959, 1.80556159176109]

plt.figure(dpi=150)
plt.loglog(resolutions, smoothed_relative_errors, 'bo-', label='With Pixel Smoothing')
plt.loglog(resolutions, unsmoothed_relative_errors, 'ro-', label='Without Pixel Smoothing')
plt.grid(True,which="both",ls="-")
plt.xlabel('resolution (pixels/μm)')
plt.ylabel('relative error in MEEP calculation (compared to PyMieScatt)')
plt.legend(loc='upper right')
plt.title('Relative Error in Mie Absorption Efficiency, Comparing MEEP and PyMieScatt')
plt.tight_layout()
plt.show()
#plt.savefig("mie_absorption_error.png")
