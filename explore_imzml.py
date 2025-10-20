from pyimzml.ImzMLParser import ImzMLParser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter


imzml_path = 'C:\\Users\\tvanheme\\Desktop\\PhD\\Code\\TBM-Smoke\\data_paper\\orbitrap\\A52 CT S3-profile.imzML'

spectra = []
mzs = []
p = ImzMLParser(imzml_path)

for idx, coords in enumerate(p.coordinates):
    mz, hs = p.getspectrum(idx)
    mzs.append(mz)
    spectra.append(hs / hs.sum())

fig1, ax1 = plt.subplots(1, 1)
ax1.plot(mzs[0], spectra[0])
ax1.set_title(0)

print('Number of spectra:', len(spectra))
print('Average m/z bins:', np.mean(list(map(len, mzs))), ' +/- ', np.std(list(map(len, mzs))))

min_mz, max_mz = np.min(list(map(np.min, mzs))), np.max(list(map(np.max, mzs)))
ax1.set_xlim([min_mz, max_mz])
ax1.set_ylim([- np.max(list(map(np.max, spectra))) * 0.1, np.max(list(map(np.max, spectra))) * 1.1])
axmz = fig1.add_axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(
    ax=axmz,
    label='m/z values [Da]',
    valmin=0,
    valmax=len(mzs) - 1,
    valinit=0,
    valstep=1,
)

def update(val):
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    ax1.cla()
    ax1.plot(mzs[int(slider.val)], spectra[int(slider.val)])
    ax1.set_title(int(slider.val))
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    fig1.canvas.draw_idle()

fig1.subplots_adjust(bottom=0.25)
slider.on_changed(update)

slider.reset()    

plt.show()

