function mpc = case_wide
% CASE_WIDE  Rows carrying all optional/solver-output columns.
mpc.version = '2';
mpc.baseMVA = 100;

%% bus data (17 columns: 13 standard + lam_P lam_Q mu_Vmax mu_Vmin)
mpc.bus = [
	1	3	0	0	0	0	1	1.02345678901	-2.5	230	1	1.06	0.94	1500.123456789	0.00225	0	0.000123;
	2	1	91.25	-12.5	0.5	19.0	1	0.987654321098	-7.123456789	230	1	1.06	0.94	1534.25	-3.5e-05	12.75	0;
	3	2	0	0	0	0	2	1.01	-1.25	115	2	1.1	0.9	1499.5	1e-08	0	0;
];

%% generator data (21 columns)
mpc.gen = [
	1	72.3	27.03	300	-300	1.04	100	1	250	10	0	0	0	0	0	0	0	0	0	0	0;
	3	163	6.54	300	-300	1.025	100	1	300	10	1.5	2.5	-10	10	-5	5	12.3456789	30	60	7.5	0.33;
];

%% branch data (17 columns: 13 standard + PF QF PT QT)
mpc.branch = [
	1	2	0.0001	0.0576	0	250	250	250	0	0	1	-360	360	71.9472	24.069	-71.9472	-20.7531;
	2	3	0.017	0.092	0.158	250	250	250	0.978	-3.5	1	-360	360	30.7281	-0.5859	-30.5547	-13.6876;
];

%% generator cost data
mpc.gencost = [
	2	1500	0	3	0.11	5	150;
	2	2000	0	3	0.085	1.2	600;
];
