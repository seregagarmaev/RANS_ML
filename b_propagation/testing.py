from modules.utils import read_field


field = 'turbulenceProperties:R'
step = 40000
path = '3_pc_Re_tau_550'

result = read_field(field,
                   step,
                   path)

print(result)
