import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

ds = xr.open_dataset('D:\Graduate-Deisgn\oisst-avhrr-v02r01.20250519.nc')

# 选时间
sst = ds['sst'].sel(time='2025-05-19', method='nearest')

# 关键：去掉 zlev 维度（表层）
sst = sst.isel(zlev=0)

print(sst.dims)    # ('lat', 'lon')
print(sst.shape)   # (720, 1440)

plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

sst.plot.pcolormesh(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cmap='coolwarm',
    vmin=-2,
    vmax=30,
    add_colorbar=True
)


ax.coastlines()
plt.title('SST 2025-05-19', fontsize=16)
plt.savefig('slide01_cover_sst.png', dpi=200, bbox_inches='tight')
plt.show()
