import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'size': 16}
matplotlib.rc('font', **font)

network_speed = np.array([2, 4, 6, 8])

rs_gzip_001_throughput = np.array([3.5572464480286126, 4.978023168589821, 5.742555436222318, 6.220210031149864])
dt_gzip_001_throughput = np.array([6.539945515380047, 13.015652251596201, 19.4280620634859, 25.778098483077223])
rf_gzip_001_throughput = np.array([6.507124612646226, 12.88629780310525, 19.14125674604337, 25.27559448008052])
ada_gzip_001_throughput = np.array([6.412228695471107, 12.519425192611358, 15.601699813636671, 15.632624445397004])
qda_gzip_001_throughput = np.array([6.505403210390705, 12.879735640296401, 19.12691900102983, 25.25072009581148])
mlp_gzip_001_throughput = np.array([6.524621073126806, 12.955252732513154, 19.293911379495718, 25.54255614647841])
gnb_gzip_001_throughput = np.array([6.5244182250994305, 12.954528106221906, 19.292359765659224, 25.539885472438094])

rs_lz4_001_throughput = np.array([3.5223510401663463, 6.883387662439317, 10.093943456261853, 13.163903165860312])
dt_lz4_001_throughput = np.array([3.7501334386074676, 7.496273623660772, 11.238426929968492, 14.976599718777472])
rf_lz4_001_throughput = np.array([3.7397080223137618, 7.454731715869432, 11.145314673925062, 14.81169729510037])
ada_lz4_001_throughput = np.array([3.7081536121397765, 7.330388058169442, 10.86965530303417, 14.32877361347074])
qda_lz4_001_throughput = np.array([3.7386337189233907, 7.451560683150099, 11.139045122792108, 14.8013476590962])
mlp_lz4_001_throughput = np.array([3.7454423400799173, 7.476903754169478, 11.194462378094881, 14.898195766523392])
gnb_lz4_001_throughput = np.array([3.7450642352394438, 7.476203284656022, 11.19349467066992, 14.897015341337314])

rs_zstd_001_throughput = np.array([5.531551284232982, 10.565452989552458, 15.165971290053925, 19.38676331389032])
dt_zstd_001_throughput = np.array([8.503741761169564, 16.985753414264313, 25.446118145320995, 33.88491871631969])
rf_zstd_001_throughput = np.array([8.450306128609782, 16.773884658236415, 24.97356513556418, 33.05209349290227])
ada_zstd_001_throughput = np.array([8.290954798936642, 15.737022153827041, 15.785648449793579, 15.810074461032563])
qda_zstd_001_throughput = np.array([8.496873594371733, 16.860379179048433, 25.093632344566615, 33.199652387493764])
mlp_zstd_001_throughput = np.array([8.491720186162729, 16.910592584773077, 25.25755059941245, 33.533511755129666])
gnb_zstd_001_throughput = np.array([8.497774102702612, 16.92270758869546, 25.275733016200295, 33.55776709204004])

fig, axs = plt.subplots(1, 3, figsize=(14, 5))
plt.subplot(131)

plt.plot(range(len(network_speed)), rs_gzip_001_throughput, label='Random Split', linestyle='dashed', marker='o')
plt.plot(range(len(network_speed)), dt_gzip_001_throughput, label='Decision Tree', marker='o')
plt.plot(range(len(network_speed)), rf_gzip_001_throughput, label='Random Forest', marker='o')
plt.plot(range(len(network_speed)), ada_gzip_001_throughput, label='AdaBoost', marker='o')
plt.plot(range(len(network_speed)), qda_gzip_001_throughput, label='QDA', marker='o')
plt.plot(range(len(network_speed)), mlp_gzip_001_throughput, label='MLP', marker='o')
plt.plot(range(len(network_speed)), gnb_gzip_001_throughput, label='Gaussian NB', marker='o')
plt.xlabel('Network Speed (MB/s)')
plt.ylabel('Throughput (MB/s)')
plt.xticks(range(len(network_speed)), network_speed)
plt.title('Gzip')

plt.legend(bbox_to_anchor=(2.8, 1.6), ncol=3)

plt.subplot(132)
plt.plot(range(len(network_speed)), rs_lz4_001_throughput, label='Random Split', linestyle='dashed', marker='o')
plt.plot(range(len(network_speed)), dt_lz4_001_throughput, label='Decision Tree', marker='o')
plt.plot(range(len(network_speed)), rf_lz4_001_throughput, label='Random Forest', marker='o')
plt.plot(range(len(network_speed)), ada_lz4_001_throughput, label='AdaBoost', marker='o')
plt.plot(range(len(network_speed)), qda_lz4_001_throughput, label='QDA', marker='o')
plt.plot(range(len(network_speed)), mlp_lz4_001_throughput, label='MLP', marker='o')
plt.plot(range(len(network_speed)), gnb_lz4_001_throughput, label='Gaussian NB', marker='o')
plt.xlabel('Network Speed (MB/s)')
plt.ylabel('Throughput (MB/s)')
plt.xticks(range(len(network_speed)), network_speed)
plt.title('LZ4')

plt.subplot(133)
plt.plot(range(len(network_speed)), rs_zstd_001_throughput, label='Random Split', linestyle='dashed', marker='o')
plt.plot(range(len(network_speed)), dt_zstd_001_throughput, label='Decision Tree', marker='o')
plt.plot(range(len(network_speed)), rf_zstd_001_throughput, label='Random Forest', marker='o')
plt.plot(range(len(network_speed)), ada_zstd_001_throughput, label='AdaBoost', marker='o')
plt.plot(range(len(network_speed)), qda_zstd_001_throughput, label='QDA', marker='o')
plt.plot(range(len(network_speed)), mlp_zstd_001_throughput, label='MLP', marker='o')
plt.plot(range(len(network_speed)), gnb_zstd_001_throughput, label='Gaussian NB', marker='o')
plt.xlabel('Network Speed (MB/s)')
plt.ylabel('Throughput (MB/s)')
plt.xticks(range(len(network_speed)), network_speed)
plt.title('Zstandard')

plt.show()

