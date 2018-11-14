"""
Small module to manage units and dimension analysis
"""
import quantities as pq


temperature = 300 * pq.K
kT = pq.UnitQuantity("k_B T", definition=300 * pq.K * 1 * pq.constants.k, symbol="kT")

length = pq.um
force = pq.UnitQuantity("nanonewton", definition=1e-9 * pq.N, symbol="nN")
energy = pq.UnitQuantity("femtojoules", definition=force * length, symbol="fJ")
time = pq.s

line_tension = energy / length
line_elasticity = energy / length ** 2
line_viscosity = force * time / length


area = length ** 2
area_tension = energy / area
area_elasticity = energy / area ** 2

vol = length ** 3
vol_tension = energy / vol
vol_elasticity = energy / vol ** 2
