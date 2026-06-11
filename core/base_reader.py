import os

import shtRipper
import matplotlib.pyplot as plt

from config.path import source_dir

fontsize=20
labelsize=20

source_dir = '../data/raw/'

# filename = 'sht46358.SHT'
# filename = 'sht42465.SHT'
# filename = 'sht37622.SHT'
filename = ('/sht39627.SHT')
# filename = '/sht46358.SHT'

# res = shtRipper.ripper.read(source_dir + filename, ['SXR', 'Ip внутр'])
# res2 = shtRipper.ripper.read(source_dir + '/sht39627.SHT', ['SXR', 'Ip внутр'])
# res3 = shtRipper.ripper.read(source_dir + '/sht46345.SHT', ['SXR', 'Ip внутр'])
# res4 = shtRipper.ripper.read(source_dir + '/sht37804.SHT', ['SXR', 'Ip внутр'])
#
# # print(res.keys())
# # print(len(res['SXR 50 mkm']['x']), len(res['SXR 50 mkm']['y']))
# # print( res['SXR 50 mkm']['x'], "\n\n", res['SXR 50 mkm']['y'])
#
# plt.subplot(2, 1, 1)
# plt.plot(res['SXR 50 mkm']['x'], res['SXR 50 mkm']['y'], color='k')
# plt.grid(True)
# # plt.title('Классическая пила', fontsize=fontsize)
# plt.xlabel('Время (с)', fontsize=fontsize)
# plt.ylabel('Сигнал', fontsize=fontsize)
# plt.tick_params(axis='both', labelsize=labelsize)
#
# plt.subplot(2, 1, 2)
# plt.plot(res['SXR 50 mkm']['x'], res['SXR 50 mkm']['y'], color='k')
# plt.grid(True)
# # plt.title('Классическая пила', fontsize=fontsize)
# plt.xlabel('Время (с)', fontsize=fontsize)
# plt.ylabel('Сигнал', fontsize=fontsize)
# plt.tick_params(axis='both', labelsize=labelsize)



# plt.subplot(2, 2, 1)
# plt.plot(res['SXR 50 mkm']['x'], res['SXR 50 mkm']['y'], color='k')
# plt.grid(True)
# plt.title('Классическая пила', fontsize=fontsize)
# plt.xlabel('Время (с)', fontsize=fontsize)
# plt.ylabel('Сигнал', fontsize=fontsize)
# plt.tick_params(axis='both', labelsize=labelsize)
#
# # plt.subplot(2, 2, 2)
# # plt.plot(res['Ip внутр.(Пр2ВК) (инт.18)']['x'], res['Ip внутр.(Пр2ВК) (инт.18)']['y'])
# # plt.grid(True)
# # plt.title('Ip внутр.(Пр2ВК) (инт.18)', fontsize=12)
# # plt.xlabel('x', fontsize=12)
# # plt.ylabel('y', fontsize=12)
# # plt.tick_params(axis='both', labelsize=12)
#
# plt.subplot(2, 2, 3)
# plt.plot(res2['SXR 15 мкм']['x'], res2['SXR 15 мкм']['y'], color='k')
# plt.grid(True)
# plt.title('Резкий первый срыв', fontsize=fontsize)
# plt.xlabel('Время (с)', fontsize=fontsize)
# plt.ylabel('Сигнал', fontsize=fontsize)
# plt.tick_params(axis='both', labelsize=labelsize)
#
# plt.subplot(2, 2, 2)
# plt.plot(res3['SXR 50 mkm']['x'], res3['SXR 50 mkm']['y'], color='k')
# plt.grid(True)
# plt.title('Смазанная пила', fontsize=fontsize)
# plt.xlabel('Время (с)', fontsize=fontsize)
# plt.ylabel('Сигнал', fontsize=fontsize)
# plt.tick_params(axis='both', labelsize=labelsize)
#
# plt.subplot(2, 2, 4)
# plt.plot(res4['SXR 50 mkm']['x'], res4['SXR 50 mkm']['y'], color='k')
# plt.grid(True)
# plt.title('Нет пилы', fontsize=fontsize)
# plt.xlabel('Время (с)', fontsize=fontsize)
# plt.ylabel('Сигнал', fontsize=fontsize)
# plt.tick_params(axis='both', labelsize=labelsize)

#
# plt.tight_layout()
# # plt.savefig(f'../data/plots/{filename}.png')
# plt.show()

for filename in os.listdir(source_dir):

    print(filename)

    # res = shtRipper.ripper.read(source_dir + filename, ['Лазер', 'SXR', 'Ip внутр'])
    res = shtRipper.ripper.read(source_dir + filename, ['SXR'])

    print(res.keys())
    # print((res['Лазер'].keys()))
    # print(len(res['Лазер']['x']),len(res['Лазер']['y']))

    keys = list(res.keys())
    # , 'Ip внутр.(Пр2ВК) (инт.18)'
    # keys = list(['SXR 80 mkm', 'SXR 50 mkm', 'SXR 15 мкм', 'SXR 127 мкм'])
    n_plots = len(keys)

    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 3*n_plots))

    for i, key in enumerate(keys):
        axes[i].plot(res[key]['x'], res[key]['y'])
        axes[i].grid(True)
        axes[i].set_title(key, fontsize=fontsize)
        axes[i].set_xlabel('x', fontsize=fontsize)
        axes[i].set_ylabel('y', fontsize=fontsize)
        axes[i].tick_params(axis='both', labelsize=labelsize)

    plt.tight_layout()
    # plt.savefig(f'../data/plots/{filename}.png')
    plt.show()

