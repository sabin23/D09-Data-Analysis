import numpy as np
import matplotlib.pyplot as plt

kpe_arr_nm= [[2.8771125251465435, 2.8790557020184577, 4.837154411149427, 3.7098484717555795, 3.9813562261254267, 1.653995679178979], [0.56783340515677, 0.3424544425098155, 0.4299534939077102, 0.6038180862079098, 0.6240230667739324, 0.4868435386636254], [0.11750608051828498, 0.04270946443817067, 0.08617193343092897, 0.0855978177971708, 0.17464478327850835, 0.06404180621186364]]
t_lead_arr_nm= [[0.4585521450513841, 0.6309058120395588, 0.4011444252914127, 0.4482261071669008, 0.48677188466705035, 0.29473831372548753], [0.18361491388726237, 0.22112320664443885, 0.19701441388886784, 0.10821847047564412, 0.14066375400426784, 0.12595527339527246], [1.4481642148512701, 2.466526636821888, 1.6078285205365332, 1.256124386520281, 0.6365799754584573, 1.4168210187693744]]
t_lag_arr_nm= [[1.9149905605253275, 2.4244507835791196, 3.136791465237117, 2.169529030240703, 1.675432786085839, 0.8727672181132089], [0.030117727091649255, 0.06544732542347359, 0.02315956437624763, 0.09111147652794482, 0.060200507252160945, 0.06686323046333394], [0.1642371725301551, 0.04431854094494592, 0.20574517667213066, 0.10724114253175077, 0.0024649232225700656, 0.010629996500568884]]
tau_arr_nm= [[0.28724298249065583, 0.23613026274847676, 0.25040692905646955, 0.25780034878378955, 0.2317304219432983, 0.32547974974823457], [0.2813032638802099, 0.2730154455027203, 0.3167691026850137, 0.24216239491351932, 0.29545042281003564, 0.3157541584661394], [0.24496814273327072, 0.3130029940729515, 0.2731260264404732, 0.2675561120692529, 0.31306992597407934, 0.3396844431560143]]
omega_nm_arr_nm= [[28.443910487943477, 11.625111939897673, 18.29611204579318, 23.120626227316308, 19.02926981128696, 19.24618137823382], [12.350255650700953, 11.386703776556828, 17.22163165658385, 16.801693609648147, 13.0442611927541, 15.084582582142787], [13.387509455582473, 12.006530054326465, 16.356909812484027, 16.93935151692527, 12.166385948964919, 13.016442477526901]]
zeta_nm_arr_nm= [[0.3147917149408267, 0.4462932909310652, 0.4689527245099856, 0.4921811347652509, 0.8838722321955284, 0.3560975244288196], [0.2426525366728769, 0.3479418150819764, 0.9966585403172052, 0.324345136588146, 0.2855239999612096, 0.26842238863890466], [0.161596612238182, 0.4970797423848204, 0.5271479028130867, 0.47030097743632604, 0.40296947909293557, 0.3362805050319589]]

kpe_arr_m= [[2.704063110968163, 2.8702436417266366, 1.8216461741024736, 1.6559193208015053, 7.431016864283766, 2.687842913947984], [0.52777328061405, 0.3369651906238369, 0.43978225479986277, 0.5684167044036585, 0.6326249983406658, 0.5006158785678145], [0.08472711466248509, 0.040945959796459375, 0.06430462813876575, 0.151391620376973, 0.20148445155607111, 0.05778053365354319]]
t_lead_arr_m= [[0.18033338356006617, 0.5117565672810818, 0.3742914290751719, 0.1800431339844336, 0.8106519012306446, 0.42832800673953497], [0.18934013647386816, 0.21811729850556655, 0.14869285601861515, 0.229011505044562, 0.07504066844862348, 0.06833675016713936], [1.6072222987750608, 2.793502299798199, 2.6235134110759395, 0.5723925749203216, 0.48791987791657276, 1.728171488606727]]
t_lag_arr_m= [[1.34216517990071, 2.2291762863279247, 1.1041958849837985, 0.5563344107326131, 4.7800403458242045, 1.838792417249988], [2.5793141951012644e-06, 7.616847273680126e-11, 0.032470244663905044, 4.33534542450931e-11, 0.04365969551086481, 0.04509382740424106], [3.335308489924534e-05, 0.004861515346107986, 0.3139837837796362, 2.6023561332290464e-09, 0.0004568397974616124, 0.06754794572177925]]
tau_arr_m= [[0.15084866201176578, 0.19312279587455933, 0.2600425755531942, 0.22201116941107643, 0.20462350667246965, 0.2702453139569735], [0.32494880817495264, 0.31682167826138763, 0.30132505016769584, 0.2897208295674898, 0.2469600709176095, 0.2637558039969705], [0.31828922423335915, 0.3714658606944474, 0.23682712864004815, 0.27897856306866675, 0.30641301286916933, 0.30883465853131636]]
omega_nm_arr_m= [[10.451618061370418, 9.010727957921409, 18.448719646332478, 16.203158576295213, 13.297118894413487, 12.976194438467765], [13.104726695792593, 9.949501819604784, 16.62636893292235, 12.486535980934958, 11.367572864922863, 12.633066011635918], [11.911524496590555, 14.381387739758608, 12.27238056938214, 11.06762600770459, 11.28210182477019, 13.851680784149735]]
zeta_nm_arr_m= [[0.2546189194210725, 0.33130394224103477, 0.45031261391662614, 0.263361687212305, 0.7279385346405083, 0.31237714342516254], [0.3459590394600991, 0.4012149076835012, 0.7168481944063203, 0.8140312572830857, 0.21996256062599712, 0.27997417039387495], [0.2751820813521556, 0.9822253400854368, 0.3699332065723116, 0.5242665578990113, 0.2515939453113316, 0.25391852534359494]]
K_m_arr_m= [[0.12207117722292776, 0.29960039519388193, 0.20645575959057189, 0.09713418382061888, 0.6881576689906153, 0.2517912420882955], [4.8937451943383903e-05, 0.03516442554882873, 0.0021321248461377983, 0.2289392295719449, 0.3112818578490673, 0.30500540519896974], [0.39630980639467506, 0.3231773317956351, 0.335198661681665, 0.44195380123782557, 0.7169607448840062, 0.5061704261375372]]
tau_m_arr_m= [[0.20312171530230466, 0.1934203386590977, 0.29037605978603864, 0.29011978897486185, 0.22504875830409554, 0.3128661414861914], [0.10248783165462205, 0.10379075604826463, 0.021734442063940037, 0.2649949321140137, 0.2497977525515947, 0.2982436254887023], [0.25600167240788196, 5.544503316684264e-08, 0.255944808447658, 0.21587428062110525, 0.22833910834622112, 0.26059610289749185]]

# make a boxplot of kpe_arr_nm values next to kpe_arr_m values for comparison

fig, ax = plt.subplots()
kpe_arr = [kpe_arr_nm[0], kpe_arr_m[0], kpe_arr_nm[1], kpe_arr_m[1], kpe_arr_nm[2], kpe_arr_m[2]]
t_lead_arr = [t_lead_arr_nm[0], t_lead_arr_m[0], t_lead_arr_nm[1], t_lead_arr_m[1], t_lead_arr_nm[2], t_lead_arr_m[2]]
t_lag_arr = [t_lag_arr_nm[0], t_lag_arr_m[0], t_lag_arr_nm[1], t_lag_arr_m[1], t_lag_arr_nm[2], t_lag_arr_m[2]]
tau_arr = [tau_arr_nm[0], tau_arr_m[0], tau_arr_nm[1], tau_arr_m[1], tau_arr_nm[2], tau_arr_m[2]]
omega_nm_arr = [omega_nm_arr_nm[0], omega_nm_arr_m[0], omega_nm_arr_nm[1], omega_nm_arr_m[1], omega_nm_arr_nm[2], omega_nm_arr_m[2]]
zeta_nm_arr = [zeta_nm_arr_nm[0], zeta_nm_arr_m[0], zeta_nm_arr_nm[1], zeta_nm_arr_m[1], zeta_nm_arr_nm[2], zeta_nm_arr_m[2]]

# arr = [kpe_arr_m[0], K_m_arr_m[0], kpe_arr_m[1], K_m_arr_m[1], kpe_arr_m[2], K_m_arr_m[2]]
arr = [tau_arr_m[0], tau_m_arr_m[0], tau_arr_m[1], tau_m_arr_m[1], tau_arr_m[2], tau_m_arr_m[2]]
ax.boxplot(arr, showfliers=False)

ax.set_title('Tau [s]')
ax.set_xticklabels(['tau C4', 'tau_m C4', 'tau C5', 'tau_m C5', 'tau C6', 'tau_m C6'])
#ax.set_yscale('log')
ax.grid()
plt.tight_layout()
plt.show()

# now plot all of these together in a 3x2 grid
fig, axs = plt.subplots(3, 2)
axs[0, 0].boxplot(kpe_arr, showfliers=False)
axs[0, 0].set_title('K_pe [s]')
axs[0, 0].set_xticklabels(['C1', 'C4', 'C2', 'C5', 'C3', 'C6'])
#axs[0, 0].set_yscale('log')
axs[0, 1].boxplot(t_lead_arr, showfliers=False)
axs[0, 1].set_title('t_lead [s]')
axs[0, 1].set_xticklabels(['C1', 'C4', 'C2', 'C5', 'C3', 'C6'])
#axs[0, 1].set_yscale('log')
axs[1, 0].boxplot(t_lag_arr, showfliers=False)
axs[1, 0].set_title('t_lag [s]')
axs[1, 0].set_xticklabels(['C1', 'C4', 'C2', 'C5', 'C3', 'C6'])
#axs[0, 2].set_yscale('log')
axs[1, 1].boxplot(tau_arr, showfliers=False)
axs[1, 1].set_title('tau [s]')
axs[1, 1].set_xticklabels(['C1', 'C4', 'C2', 'C5', 'C3', 'C6'])
#axs[0, 3].set_yscale('log')
axs[2, 0].boxplot(omega_nm_arr, showfliers=False)
axs[2, 0].set_title('omega_nm [rad/s]')
axs[2, 0].set_xticklabels(['C1', 'C4', 'C2', 'C5', 'C3', 'C6'])
#axs[1, 0].set_yscale('log')
axs[2, 1].boxplot(zeta_nm_arr, showfliers=False)
axs[2, 1].set_title('zeta_nm [-]')
axs[2, 1].set_xticklabels(['C1', 'C4', 'C2', 'C5', 'C3', 'C6'])
#axs[1, 1].set_yscale('log')
plt.tight_layout()
#plt.show()