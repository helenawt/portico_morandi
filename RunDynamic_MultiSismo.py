"""
RunDynamic_MultiSismo.py
========================
Análise Dinâmica Não-Linear para MÚLTIPLOS SISMOS.

Estratégia:
  - O modelo OpenSees é reconstruído do zero para cada terremoto (ops.wipe obrigatório
    porque os materiais não-lineares acumulam dano entre análises).
  - Os resultados de todos os sismos são acumulados em ResultadosFinais e salvos num
    único CSV consolidado ao final.
  - Um CSV parcial é gravado após cada sismo bem-sucedido (segurança contra crashes).
  - Sismos que falharam registram zeros mas são igualmente incluídos no CSV final.

Configuração rápida (seção "PARÂMETROS DO USUÁRIO"):
  - pasta_terremotos  : pasta com os arquivos .at2
  - filtro_terremotos : lista explícita de arquivos OU None para usar todos da pasta
  - escala_terremoto  : fator de escala (float ou None → 1.0)
  - parametros        : dicionário de listas para varredura de dt_fatorreducao / tolerancia / maxiter
"""

# =============================================================================
# Imports
# =============================================================================
import os
import time
import shutil
import pickle
from datetime import datetime

import openseespy.opensees as ops
import opsvis as opsv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Geometry_Materials_atualizado_certo as geom
# import VerticalLoad as load
# import Mass_ok as mass

b_col = 0.35 #m
h_col = 0.35  #m
b_beam = b_col
h_beam = h_col

X_spans = [0.0, 4.57]   # Spans in X direction [min=2] [max=9]  
Z_pavs = [0.0, 3.125]      # Floor levels [min=2] [max=99]

Mid_discretiza_Col = 3   # Mesh size for the column middle zones   
Extremos_Inf_Discretiza_Col = 1 # Mesh size for the top end zones
Extremos_Sup_Discretiza_Col = 1   # Mesh size for the bottom end zones

Extremos_Discretiza_Beam = 1 # Mesh size for the beam end zones
Internos_Discretiza_Beam = 1   # Mesh size for the adjacent zones to the beam ends
Mid_discretiza_Beam = 2 # Mesh size for the beam middle zones

fc_normal = 34 #MPa
Es_normal = 200 # 201 GPa  # Modulo de elasticidade do aço
fy_normal_Column_TopBars = 491 #22mm

massPav =  [0.0, 2147.3224999999998] #veio do Mass_ok

base_dir = os.path.dirname(__file__)

# =============================================================================
# PARÂMETROS DO USUÁRIO  ← edite aqui
# =============================================================================

# Pega a pasta onde o seu script atual está salvo
diretorio_atual = os.path.dirname(os.path.abspath(__file__))

# Constrói o caminho a partir dali
pasta_terremotos = os.path.join(diretorio_atual, 'AT2_por_faixa_PGAgeom', '0.20-0.30')
# pasta_terremotos = './AT2_por_faixa_PGAgeom/0.20-0.30'
# pasta_terremotos = './Terremotos'

# Lista explícita de sismos (nomes dos arquivos .at2).
# Use None para rodar TODOS os arquivos .at2 encontrados na pasta.
filtro_terremotos = None
# filtro_terremotos = ['RSN6_IMPVALL.I_I-ELC180.AT2', 'RSN31_PARKF_C08050.AT2', 'RSN31_PARKF_C08320.AT2']
# filtro_terremotos = ['RSN1_HELENA.A_A-HMC180.at2']
# filtro_terremotos = ['RSN1007_NORTHR_UNI005.at2', 'RSN0900_LANDERS_LCN260.at2']


# Fator de escala aplicado a todos os sismos (None → 1.0)
escala_terremoto = 1

# Cortar sinal? (1 = cortar DS5-95, None = sinal completo)
cortar_sinal = 1

# Salvar figuras de espectro / sinal por sismo?
salvar_figuras = True

# Amortecimento de Rayleigh
eps_amortecimento = 0.05

# Número de modos para análise modal
numero_de_modos = 3
# Nota: a discretização de colunas (Extremos_Inf/Mid/Extremos_Sup_Discretiza_Col)
# e de vigas (Extremos/Internos/Mid_discretiza_Beam) é definida em Geometry_Materials.py

# Parâmetros numéricos – aceita listas para varredura automática
parametros = {
    'dt_fatorreducao': [1],
    'tolerancia':      [1e-4],
    'maxiter':         [1000],
}

# Infill / offset (repassados para Geometry_Materials.py via globals)
Infill_wall = 'No'
offset      = 'No'

# =============================================================================
# Funções auxiliares (copiadas / refatoradas do RunDynamic.py original)
# =============================================================================

def ReadRecord_Interpolado(terremoto, dt_fatorreducao=1, pasta_data=None):
    with open(terremoto, 'r') as f:
        lines = f.readlines()
    header_line = lines[3]
    parts   = header_line.split(',')
    npts    = int(parts[0].strip().split('=')[1])
    dt      = float(parts[1].strip().split('=')[1].split()[0])

    data_str            = " ".join(lines[4:])
    aceleracao_original = np.array([float(v) for v in data_str.split()])
    aceleracao_original = np.insert(aceleracao_original, 0, 0.0)

    from decimal import Decimal
    casas = abs(Decimal(str(dt)).as_tuple().exponent)
    t_original      = np.arange(1, npts + 1) * dt
    t_original      = np.insert(t_original, 0, 0.0)
    t_original[-1]  = round(t_original[-1], casas)

    dt_novo = dt / dt_fatorreducao
    t_novo  = np.arange(dt_novo, round(t_original[-1] + dt_novo, casas), dt_novo)
    aceleracao_novo = np.interp(t_novo, t_original, aceleracao_original)
    npts_novo = len(t_novo)

    if pasta_data is not None:
        with open(pasta_data, 'w') as g:
            for i in range(0, len(aceleracao_novo), 5):
                linha = "  ".join(f"{aceleracao_novo[j]:.7E}"
                                  for j in range(i, min(i + 5, len(aceleracao_novo))))
                g.write(linha + "\n")

    return dt_novo, npts_novo


def ReadRecord(inFilename, outFilename):
    dt = 0.0; npts = 0
    inFileID  = open(inFilename,  'r')
    outFileID = open(outFilename, 'w')
    flag = 0
    for line in inFileID:
        if line == '\n':
            continue
        elif flag == 1:
            outFileID.write(line)
        else:
            words = line.split()
            if len(words) >= 4:
                if words[0] == 'NPTS=':
                    for word in words:
                        if word != '':
                            if flag == 1:
                                dt = float(word); break
                            if flag == 2:
                                npts = int(word.strip(',')); flag = 0
                            if word == 'DT=' or word == 'dt':
                                flag = 1
                            if word == 'NPTS=':
                                flag = 2
                elif words[-1] == 'DT':
                    count = 0
                    for word in words:
                        if word != '':
                            if count == 0:   npts = int(word)
                            elif count == 1: dt   = float(word)
                            elif word == 'DT':
                                flag = 1; break
                            count += 1
    inFileID.close(); outFileID.close()
    return dt, npts


def criar_pasta(nomepasta):
    os.makedirs(nomepasta, exist_ok=True)


def modelo2D_pastas(terremoto):
    base = f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{os.path.splitext(terremoto)[0]}_ODB'
    criar_pasta(base)
    criar_pasta(base + '/Figuras')
    criar_pasta(base + '/Resultados')
    criar_pasta(base + '/Sinais_DAT')


def modelo2D_preparando_earthquake(terremoto, pasta_terremotos, escala_terremoto,
                                   dt_fatorreducao, num_pav):
    print(f"\nAplicação do Terremoto: {terremoto}\n")
    tag_ts = 1; tag_pat = 1
    nome    = os.path.splitext(terremoto)[0]
    g       = 9.80665

    dt, nPts = ReadRecord_Interpolado(
        f'{pasta_terremotos}/{nome}.at2',
        dt_fatorreducao=dt_fatorreducao,
        pasta_data=f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Sinais_DAT/{nome}.dat'
    )
    ops.timeSeries('Path', tag_ts + 1,
                   '-filePath', f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Sinais_DAT/{nome}.dat',
                   '-dt', dt, '-factor', g * escala_terremoto, '-prependZero')
    ops.pattern('UniformExcitation', tag_pat + 1, 1, '-accel', tag_ts + 1)

    listapav = [int(str(ii) + '01') for ii in range(1, num_pav + 1)]

    ops.recorder('Node', '-file',
                 f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados/{nome}_Disp.out',
                 '-time', '-node', *listapav, '-dof', 1, 2, 3, 'disp')
    ops.recorder('Node', '-file',
                 f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados/{nome}_AccelsREL.out',
                 '-time', '-node', *listapav, '-dof', 1, 2, 3, 'accel')
    ops.recorder('Node', '-file',
                 f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados/{nome}_AccelsABS_x.out',
                 '-timeSeries', tag_ts + 1,
                 '-node', *listapav, '-dof', 1, 'accel')
    ops.recorder('Node', '-file',
                 f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados/{nome}_Vel.out',
                 '-time', '-node', *listapav, '-dof', 1, 2, 3, 'vel')


def modelo2D_espectro(terremoto, pasta_terremotos, escala_terremoto,
                      T1_static, salvar_figuras=False, diretorio=None):
    from matplotlib.ticker import ScalarFormatter
    import pyrotd
    pyrotd.processes = 1

    nome = os.path.splitext(terremoto)[0]
    fname_x = os.path.join(pasta_terremotos, nome + '.at2')
    with open(fname_x) as f:
        for _ in range(3):
            next(f)
        next(f)
        accels_x = np.array([p for l in f for p in l.split()]).astype(float)

    dt, nPts = ReadRecord_Interpolado(fname_x, dt_fatorreducao=1)
    accels_x = np.insert(accels_x, 0, 0.0)
    if escala_terremoto is not None:
        accels_x = accels_x * escala_terremoto

    from scipy.integrate import cumulative_trapezoid
    tempo_sismo = np.insert(np.linspace(dt, dt * nPts, nPts), 0, 0.0)
    vel_history   = cumulative_trapezoid(accels_x * 9.80665, tempo_sismo, initial=0)
    desloc_history= cumulative_trapezoid(vel_history, tempo_sismo, initial=0)

    osc_freqs    = np.logspace(-1, 2, 100)
    eps_spectro  = 0.05
    resp_spec_x  = pyrotd.calc_spec_accels(dt, accels_x, osc_freqs, eps_spectro, osc_type='psa')
    Periodos_x   = 1 / resp_spec_x.osc_freq
    Sa_x_Aceleracoes = resp_spec_x.spec_accel

    resp_Sa = pyrotd.calc_spec_accels(dt, accels_x,
                                      [1/0.2, 1/0.3, 1/0.6, 1/1, 1/T1_static],
                                      eps_spectro, osc_type='psa')
    resp_Sv = pyrotd.calc_spec_accels(dt, accels_x, [1/T1_static], eps_spectro, osc_type='psv')
    resp_Sd = pyrotd.calc_spec_accels(dt, accels_x, [1/T1_static], eps_spectro, osc_type='sd')

    Sa_T_02s = round(resp_Sa.spec_accel[0], 2)
    Sa_T_03s = round(resp_Sa.spec_accel[1], 2)
    Sa_T_06s = round(resp_Sa.spec_accel[2], 2)
    Sa_T_1s  = round(resp_Sa.spec_accel[3], 2)
    Sa_T1    = round(resp_Sa.spec_accel[4], 2)
    Sv_T1    = resp_Sv.spec_accel[0]
    Sd_T1    = resp_Sd.spec_accel[0]
    PGA      = round(max(abs(accels_x)), 4)
    PGV      = round(max(abs(vel_history)), 6)
    PGD      = round(max(abs(desloc_history)), 6)

    if salvar_figuras:
        _save_fig_sismo(nome, tempo_sismo, accels_x, vel_history, desloc_history,
                        Periodos_x, Sa_x_Aceleracoes, eps_spectro, escala_terremoto, diretorio)

    return (tempo_sismo, accels_x, vel_history, desloc_history,
            Periodos_x, Sa_x_Aceleracoes,
            Sa_T_02s, Sa_T_03s, Sa_T_06s, Sa_T_1s, Sa_T1, Sv_T1, Sd_T1,
            PGA, PGV, PGD)


def _save_fig_sismo(nome, tempo, accels, vel, desloc,
                    Periodos, Sa_Aceleracoes, eps_spectro, escala, diretorio):
    from matplotlib.ticker import ScalarFormatter
    def _salvar(fname):
        plt.savefig(fname if diretorio is None else f'{diretorio}/{fname}')
        plt.close()

    plt.figure(); plt.plot(tempo, accels, color='black')
    plt.xlabel("Time (s)"); plt.ylabel("Acceleration (g)"); plt.grid()
    plt.title('Acceleration'); plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    _salvar(nome + '_SinalACCELS.png')

    plt.figure(); plt.plot(tempo, desloc, color='black')
    plt.xlabel("tempo (s)"); plt.ylabel("deslocamento (m)"); plt.grid()
    plt.title('Histórico de Deslocamento'); plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    _salvar(nome + '_SinalDESLOC.png')

    plt.figure(); plt.plot(tempo, vel, color='black')
    plt.xlabel("tempo (s)"); plt.ylabel("velocidade (m/s)"); plt.grid()
    plt.title('Histórico de Velocidade'); plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    _salvar(nome + '_SinalVEL.png')

    plt.figure(); plt.plot(Periodos, Sa_Aceleracoes, color='blue')
    plt.xlabel("Period (s)"); plt.ylabel("Spectral Acceleration (g)"); plt.grid()
    plt.title(f'Response Spectrum Sa, ξ={eps_spectro*100:.0f}%')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())
    _salvar(nome + '_Spectrum_Sa.png')


def cortar_sismos(terremoto_dat_x, terremoto_dat_y=None, dt=0.01, margem=2.0, editar_dat=False):
    if terremoto_dat_y is None or terremoto_dat_y == terremoto_dat_x:
        terremoto_dat_y = terremoto_dat_x

    def _read_dat(path):
        with open(path) as f:
            lines = f.readlines()
        acc = np.array([float(v) for v in " ".join(lines).split()])
        return np.insert(acc, 0, 0.0)

    def calcular_ds5_95(acc):
        acc_si = acc * 9.80665
        ia = np.cumsum(acc_si ** 2) * (np.pi / (2 * 9.80665)) * dt
        ia /= ia[-1]
        return np.argmax(ia >= 0.025), np.argmax(ia >= 0.975)

    acc_x = _read_dat(terremoto_dat_x)
    acc_y = _read_dat(terremoto_dat_y)
    t5_x, t95_x = calcular_ds5_95(acc_x)
    t5_y, t95_y = calcular_ds5_95(acc_y)
    idx_ini = max(0, min(t5_x, t5_y) - int(margem / dt))
    idx_fim = min(max(t95_x, t95_y) + int(margem / dt), len(acc_x))
    acc_x_c = acc_x[idx_ini:idx_fim]
    acc_y_c = acc_y[idx_ini:idx_fim]

    if editar_dat:
        def _salvar(path, arr):
            with open(path, 'w') as f:
                for i in range(0, len(arr), 5):
                    f.write("  ".join(f"{v:.8E}" for v in arr[i:i+5]) + "\n")
        _salvar(terremoto_dat_x, acc_x_c)
        _salvar(terremoto_dat_y, acc_y_c)

    return acc_x_c, acc_y_c


def analyze_earthquake2(terremoto, pasta_terremotos, diretorio_analise,
                        T1_static, tolerancia, dt_fatorreducao, max_iter, cortar_sinal):
    nome     = os.path.splitext(terremoto)[0]
    dt, nPts = ReadRecord_Interpolado(f'{pasta_terremotos}/{nome}.at2',
                                      dt_fatorreducao=dt_fatorreducao)

    if cortar_sinal == 1:
        dat_x = f'{diretorio_analise}/Sinais_DAT/{nome}.dat'
        acc_x_c, _ = cortar_sismos(dat_x, dat_x, dt=dt, margem=2.0, editar_dat=False)
        nPts = len(acc_x_c)

    tFinal   = nPts * dt
    tCurrent = ops.getTime()
    ok       = 0
    tipo_algoritmo = {1: 'NewtonLineSearch', 2: 'KrylovNewton',
                      3: 'ModifiedNewton',   4: 'BFGS'}
    falha    = 0

    ops.wipeAnalysis()
    ops.system('UmfPack')
    ops.constraints('Transformation')
    ops.numberer('RCM')

    while tCurrent < tFinal and falha == 0:
        jj = 1
        ops.test('NormDispIncr', tolerancia, max_iter, 2)
        ops.algorithm(tipo_algoritmo[jj])
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')
        ok = ops.analyze(1, dt)

        if ok != 0:
            for jj in range(2, len(tipo_algoritmo) + 1):
                ops.algorithm(tipo_algoritmo[jj])
                ok = ops.analyze(1, dt)
                if ok == 0:
                    break
            if ok != 0:
                # tenta passo reduzido
                ok = ops.analyze(10, dt / 10)
            if ok != 0:
                falha = 1

        tCurrent = ops.getTime()

    analise = 'SUCESSO' if falha == 0 else 'FALHOU'
    print(f'\nAnálise Finalizada com {analise}')
    return analise


def analyze_static():
    ops.wipeAnalysis()
    ops.system('UmfPack')
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.test('NormDispIncr', 1.0e-8, 10, 3)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 1)
    ops.analysis('Static')
    ops.analyze(1)
    ops.reactions()
    print('  Carga vertical aplicada (análise estática concluída)')


def analyze_modal(numero_de_modos=5, eps_amortecimento=0.05,
                  salvar_figuras=False, diretorio=None,
                  modal_filename="ModalReport.out"):
    try:
        eigenValues = ops.eigen('-genBandArpack', numero_de_modos)
    except Exception:
        eigenValues = ops.eigen('-fullGenLapack', numero_de_modos)

    w1 = np.sqrt(eigenValues[0])
    w3 = np.sqrt(eigenValues[2])
    T1 = 2 * np.pi / w1
    alphaM    = eps_amortecimento * (2 * w1 * w3) / (w1 + w3)
    betaKcomm = 2 * eps_amortecimento / (w1 + w3)
    ops.rayleigh(alphaM, 0, 0, betaKcomm)

    Propriedades_Modal = ops.modalProperties("-file", modal_filename, "-unorm", "-return")

    if salvar_figuras:
        for ii in range(1, 4):
            opsv.plot_mode_shape(ii, 100, node_supports=True, endDispFlag=0)
            fname = f'ModoVibracao{ii}.png'
            plt.savefig(fname if diretorio is None else f'{diretorio}/{fname}')
            plt.close()

    return Propriedades_Modal, float(T1)


def remover_analises():
    ops.remove('recorders')
    for tag in [1, 2, 3]:
        try: ops.remove('loadPattern', tag)
        except Exception: pass
        try: ops.remove('timeSeries', tag)
        except Exception: pass
    ops.reset()


def modelo2D_IDR_drift(terremoto, Z_pavs):
    nome     = os.path.splitext(terremoto)[0]
    
    # SEGURO CONTRA KEYERROR: Se Z_pavs vier como dicionário, extrai os valores ordenados em uma lista.
    # Se já for uma lista, garante que ela comece com 0.0 caso não tenha.
    if isinstance(Z_pavs, dict):
        # Transforma {1: 3.125} em [0.0, 3.125]
        valores_alturas = sorted(list(Z_pavs.values()))
        Z_list = valores_alturas


    num_pav  = len(Z_list)
    Desloc   = np.loadtxt(f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados/{nome}_Disp.out')
    
    Drift_x  = np.zeros([len(Desloc), num_pav])
    Drift_x[:, 0] = Desloc[:, 0]
    MaxIDR_x = np.zeros((1, num_pav - 1)) # Correção na inicialização do array numpy
    
    aux = 1
    for pav in range(1, num_pav):
        # Usando 'Z_list' que temos certeza que aceita os índices 1 e 0 sem dar KeyError
        Drift_x[:, pav] = (Desloc[:, aux + 3] - Desloc[:, aux]) / (Z_list[pav] - Z_list[pav - 1])
        aux += 3
        MaxIDR_x[0][pav - 1] = np.max(abs(Drift_x[:, pav]))
        
    MaxDrift_Edificio = np.max(abs(Desloc[:, -3]) / Z_list[-1])
    return MaxIDR_x, float(np.max(MaxIDR_x)), float(MaxDrift_Edificio)

def modelo2D_deslocamento(terremoto, Z_pavs):
    nome = os.path.splitext(terremoto)[0]
    Desloc = np.loadtxt(f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados/{nome}_Disp.out')
    aux = 1
    vals = []
    for _ in range(1, len(Z_pavs) + 1):
        vals.append(Desloc[:, aux])
        aux += 3
    return float(np.max(abs(np.column_stack(vals))))


def modelo2D_velocidade(terremoto, Z_pavs, vel_history, pasta_terremotos):
    nome    = os.path.splitext(terremoto)[0]
    num_pav = len(Z_pavs)
    Vel     = np.loadtxt(f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados/{nome}_Vel.out')
    dt, nPts = ReadRecord(f'{pasta_terremotos}/{nome}.at2', './deletar')
    time_history = np.insert(np.linspace(dt, nPts * dt, nPts), 0, 0.0)

    MaxVEL_ABS_x = np.array([[0.0] * num_pav])
    Vel_ABS      = np.zeros([len(Vel), num_pav + 1])
    Vel_ABS[:, 0] = Vel[:, 0]
    MaxVEL_Rel   = 0.0
    aux = 1
    for pav in range(1, num_pav + 1):
        vel_rel = Vel[:, aux]
        MaxVEL_Rel = max(MaxVEL_Rel, float(np.max(abs(vel_rel))))
        vel_interp = np.interp(Vel[:, 0], time_history, vel_history)
        Vel_ABS[:, pav] = vel_rel + vel_interp
        MaxVEL_ABS_x[0, pav - 1] = float(np.max(abs(Vel_ABS[:, pav])))
        aux += 3

    MaxVEL_ABS = float(np.max(abs(Vel_ABS[:, 1:])))
    return MaxVEL_ABS_x, MaxVEL_ABS, MaxVEL_Rel


def modelo2D_aceleracaoABS(terremoto, Z_pavs):
    nome    = os.path.splitext(terremoto)[0]
    num_pav = len(Z_pavs)
    Acc     = np.loadtxt(f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados/{nome}_AccelsABS_x.out')
    MaxAcc_x = np.array([[0.0] * num_pav])
    for pav in range(num_pav):
        MaxAcc_x[0][pav] = float(np.max(abs(Acc[:, pav])))
    return MaxAcc_x, float(np.max(MaxAcc_x))


def modelo2D_resultados(X_spans, Z_pavs, b_col, h_col,
                        Extremos_Inf_Discretiza_Col, Mid_discretiza_Col, Extremos_Sup_Discretiza_Col,
                        b_beam, h_beam,
                        Extremos_Discretiza_Beam, Internos_Discretiza_Beam, Mid_discretiza_Beam,
                        analise_starttime,
                        terremoto, analise, escala_terremoto,
                        MaxIDR, MaxIDR_pavs_x, MaxDrift_Edificio,
                        MaxDESLOC, MaxVEL_ABS_pavs_x, MaxVEL_ABS, MaxVEL_Rel,
                        MaxAccels_ABS_pavs_x, MaxACCEL_ABS,
                        T1_static, T1_sismo, fc_normal, fy_normal, Es_normal,
                        eps_amortecimento, massPav,
                        Sa_T1, Sa_T_02s, Sa_T_03s, Sa_T_06s, Sa_T_1s,
                        Sv_T1, Sd_T1, PGA, PGV, PGD, infill_wall, resultados_tit):
    num_pav = len(Z_pavs)
    g = 9.80665

    resultado = [
        terremoto, analise, escala_terremoto, num_pav,
        MaxIDR * 100, MaxIDR_pavs_x[0] * 100, 0, MaxIDR_pavs_x[0] * 100, MaxDrift_Edificio * 100,
        MaxDESLOC, 0, MaxDESLOC,
        MaxVEL_ABS, MaxVEL_ABS_pavs_x[0], 0, MaxVEL_ABS_pavs_x[0],
        MaxACCEL_ABS, MaxAccels_ABS_pavs_x[0], 0, MaxAccels_ABS_pavs_x[0],
        T1_static, T1_sismo, fc_normal, fy_normal, Es_normal, eps_amortecimento,
        massPav[num_pav - 1],
        Sa_T1, Sa_T_02s, Sa_T_03s, Sa_T_06s, Sa_T_1s,
        Sv_T1 * g * 100, Sd_T1 * g * 100, PGA, PGV * 100, PGD * 100,
        infill_wall
    ]
    resultado_dic = {resultados_tit[i]: resultado[i] for i in range(len(resultados_tit))}

    # Save per-earthquake result
    nome = os.path.splitext(terremoto)[0]
    resultados_dir = f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Resultados'
    os.makedirs(resultados_dir, exist_ok=True)
    pd.DataFrame([resultado_dic]).to_csv(
        f'{resultados_dir}/Result_{nome}.out', index=False)
    pd.DataFrame([resultado_dic]).to_csv(
        f'{resultados_dir}/ResultadosFinaisParcial.out', index=False)

    total_col_elem  = Extremos_Inf_Discretiza_Col + Mid_discretiza_Col + Extremos_Sup_Discretiza_Col
    total_beam_elem = 2 * Extremos_Discretiza_Beam + 2 * Internos_Discretiza_Beam + Mid_discretiza_Beam

    analise_endtime = datetime.now()
    resumo_path = f'{resultados_dir}/Resumo_{nome}.txt'
    with open(resumo_path, 'w') as f:
        f.write(f"Terremoto: {nome}\n")
        f.write(f"Análise: {analise}\n")
        f.write(f"Fator de escala: {escala_terremoto}\n")
        f.write(f"Nº Pavimentos: {num_pav}\n")
        f.write(f"\n--- Geometria ---\n")
        f.write(f"Coordenadas dos Tramos     x (m): {X_spans}\n")
        f.write(f"Coordenadas dos Pavimentos z (m): {Z_pavs}\n")
        f.write(f"Pilares - base: {b_col:.2f} m   altura: {h_col:.2f} m\n")
        f.write(f"  Discretizacao colunas  - Inf: {Extremos_Inf_Discretiza_Col}  "
                f"Meio: {Mid_discretiza_Col}  Sup: {Extremos_Sup_Discretiza_Col}  "
                f"(total: {total_col_elem} elem.)\n")
        f.write(f"Vigas   - base: {b_beam:.2f} m   altura: {h_beam:.2f} m\n")
        f.write(f"  Discretizacao vigas    - Extremos: {Extremos_Discretiza_Beam}  "
                f"Internos: {Internos_Discretiza_Beam}  Meio: {Mid_discretiza_Beam}  "
                f"(total: {total_beam_elem} elem.)\n")
        f.write(f"\n--- Resultados ---\n")
        f.write(f"T1 static: {T1_static:.5f} s\n")
        f.write(f"Max IDR: {MaxIDR * 100:.5f} %\n")
        f.write(f"Max IDR por pav.: {[f'{v*100:.5f}%' for v in MaxIDR_pavs_x[0]]}\n")
        f.write(f"Max Drift Base-Topo: {MaxDrift_Edificio * 100:.5f} %\n")
        f.write(f"Max Desloc.: {MaxDESLOC:.5f} m\n")
        f.write(f"Max Vel ABS: {MaxVEL_ABS:.5f} m/s\n")
        f.write(f"Max Acel ABS: {MaxACCEL_ABS:.5f} m/s2\n")
        f.write(f"PGA: {PGA:.5f} g\n")
        f.write(f"PGV: {PGV * 100:.5f} cm/s\n")
        f.write(f"PGD: {PGD * 100:.5f} cm\n")
        f.write(f"Sa(T1): {Sa_T1:.5f} g\n")
        f.write(f"Duracao da Analise: {analise_endtime - analise_starttime}\n")

    print(f"\n{'='*60}")
    print(f"  RESULTADO — {nome}")
    print(f"{'='*60}")
    print(f"  Análise    : {analise}")
    print(f"  Pilares    : {b_col:.2f}x{h_col:.2f} m  |  discret. Inf={Extremos_Inf_Discretiza_Col} Meio={Mid_discretiza_Col} Sup={Extremos_Sup_Discretiza_Col} (total={total_col_elem})")
    print(f"  Vigas      : {b_beam:.2f}x{h_beam:.2f} m  |  discret. Ext={Extremos_Discretiza_Beam} Int={Internos_Discretiza_Beam} Meio={Mid_discretiza_Beam} (total={total_beam_elem})")
    print(f"  Max IDR    : {MaxIDR * 100:.4f} %")
    print(f"  Max Desloc.: {MaxDESLOC:.5f} m")
    print(f"  Max Vel ABS: {MaxVEL_ABS:.5f} m/s")
    print(f"  Max Acc ABS: {MaxACCEL_ABS:.5f} m/s²")
    print(f"  Sa(T1)     : {Sa_T1:.5f} g   PGA: {PGA:.5f} g")
    print(f"  T1 static  : {T1_static:.5f} s")
    print(f"  Duração    : {analise_endtime - analise_starttime}")
    print(f"{'='*60}\n")

    return resultado_dic


# =============================================================================
# Lista de terremotos
# =============================================================================

if filtro_terremotos is not None:
    # Usa lista explícita fornecida pelo usuário
    terremotos = [t for t in filtro_terremotos
                  if os.path.isfile(os.path.join(pasta_terremotos, t))]
else:
    # Usa todos os .at2 da pasta
    terremotos = sorted([f for f in os.listdir(pasta_terremotos)
                         if f.lower().endswith('.at2')])

print(f"\n{'#'*60}")
print(f"  ANÁLISE MULTI-SISMO")
print(f"  Total de terremotos encontrados: {len(terremotos)}")
print(f"  Pasta: {os.path.abspath(pasta_terremotos)}")
print(f"{'#'*60}\n")

if len(terremotos) == 0:
    raise FileNotFoundError(
        f"Nenhum arquivo .at2 encontrado em '{pasta_terremotos}'. "
        "Verifique o caminho ou a variável filtro_terremotos."
    )

# =============================================================================
# Títulos das colunas de resultado
# =============================================================================

resultados_tit = [
    "Earthquake", "Analysis", "Scale Factor (g)", "Number of Floors",
    "Max. IDR (%)", "Max. IDRx per floor (%)", "Max. IDRy per floor (%)",
    "Max. IDR Resultant per floor (%)", "Max. Drift Base-Top Building (%)",
    "Max. Displacement in x (m)", "Max. Displacement in y (m)",
    "Max. Resultant Displacement (m)",
    "Max. Vel. (m/s)", "Max. Velx per floor (m/s)",
    "Max. Vely per floor (m/s)", "Max. Vel Resultant per floor (m/s)",
    "Max. Accel. per floor (m/s²)", "Max. Accelx per floor (m/s²)",
    "Max. Accely per floor (m/s²)", "Max. Accel Resultant per floor (m/s²)",
    "T1 Initial (s)", "T1 Post-Earthquake (s)",
    "Fc (MPa)", "Fy (MPa)", "Es (GPa)", "Damping Ratio",
    "Total Floor Mass (kg)",
    "Sa(T1) (g)", "Sa(T=0.2s) (g)", "Sa(T=0.3s) (g)",
    "Sa(T=0.6s) (g)", "Sa(T=1s) (g)",
    "Sv(T1) (cm/s)", "Sd(T1) (cm)",
    "PGA (g)", "PGV (cm/s)", "PGD (cm)",
    "Infill Wall"
]

ResultadosFinais = []
start_time = time.time()

# Garantir que a pasta principal de resultados existe
frame_folder = 'InfillFrame' if Infill_wall == 'Yes' else 'BareFrame'
os.makedirs(f'./Results_Dynamic_Multisismos/{frame_folder}', exist_ok=True)


# =============================================================================
# LOOP PRINCIPAL — um modelo por terremoto
# =============================================================================

for idx_sismo, terremoto in enumerate(terremotos):

    print(f"\n{'#'*60}")
    print(f"  SISMO {idx_sismo + 1}/{len(terremotos)} : {terremoto}")
    print(f"{'#'*60}")

    analise_starttime = datetime.now()
    nome = os.path.splitext(terremoto)[0]

    # ── Varredura de parâmetros numéricos ──────────────────────────────────
    analise = 'FALHOU'

    for dt_fatorreducao in parametros['dt_fatorreducao']:
        if analise == 'SUCESSO':
            break
        for tolerancia in parametros['tolerancia']:
            if analise == 'SUCESSO':
                break
            for max_iter in parametros['maxiter']:
                if analise == 'SUCESSO':
                    break

                # ── Reconstruir modelo do zero ─────────────────────────────
                ops.wipe()
                ops.model('basic', '-ndm', 2)

                exec(open(os.path.join(base_dir, 'Geometry_Materials_atualizado_certo.py')).read(), globals())
                exec(open(os.path.join(base_dir, 'VerticalLoad.py')).read(), globals())
                exec(open(os.path.join(base_dir, 'Mass_ok.py')).read(), globals())

                if Infill_wall == 'Yes':
                    exec(open('Geometry_InfillFrame.py').read())

                # ── Prefijo para nombres de ModalReport ────────────────────
                diretorio         = f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB/Figuras/'
                diretorio_analise = f'./Results_Dynamic_Multisismos/{frame_folder}/Modelo2D_{nome}_ODB'
                modelo2D_pastas(terremoto)

                modal_prefix = 'InfillFrame' if Infill_wall == 'Yes' else 'BareFrame'
                modal_pre  = f'{diretorio_analise}/ModalReport_{modal_prefix}_1_PreVerticalLoad.out'
                modal_post = f'{diretorio_analise}/ModalReport_{modal_prefix}_2_PostVerticalLoad.out'
                modal_dyn  = f'{diretorio_analise}/ModalReport_{modal_prefix}_3_PostDynamic.out'

                # ── Modal Report 1: antes da carga vertical ────────────────
                analyze_modal(
                    numero_de_modos=numero_de_modos,
                    eps_amortecimento=eps_amortecimento,
                    salvar_figuras=False,
                    diretorio=None,
                    modal_filename=modal_pre
                )
                print(f'  ModalReport 1 gerado (pré carga vertical): {modal_pre}')

                # ── Análise Estática Gravitacional ─────────────────────────
                analyze_static()

                # ── Modal Report 2: depois da carga vertical ───────────────
                Propriedades_Modal, T1_static = analyze_modal(
                    numero_de_modos=numero_de_modos,
                    eps_amortecimento=eps_amortecimento,
                    salvar_figuras=salvar_figuras,
                    diretorio=diretorio,
                    modal_filename=modal_post
                )
                print(f'  T1 = {T1_static:.4f} s  |  dt_red={dt_fatorreducao}  tol={tolerancia}  iter={max_iter}')
                print(f'  ModalReport 2 gerado (pós carga vertical): {modal_post}')

                esc = escala_terremoto if escala_terremoto is not None else 1.0

                # ── Espectro de Resposta ───────────────────────────────────
                (tempo_sismo, accels_x, vel_history, desloc_history,
                 Periodos_x, Sa_x_Aceleracoes,
                 Sa_T_02s, Sa_T_03s, Sa_T_06s, Sa_T_1s, Sa_T1,
                 Sv_T1, Sd_T1, PGA, PGV, PGD) = modelo2D_espectro(
                    terremoto=terremoto,
                    pasta_terremotos=pasta_terremotos,
                    escala_terremoto=esc,
                    T1_static=T1_static,
                    salvar_figuras=salvar_figuras,
                    diretorio=diretorio
                )

                # ── Congelar cargas verticais e resetar tempo ─────────────
                ops.loadConst('-time', 0.0)
                num_pav = len(Z_pavs)

                modelo2D_preparando_earthquake(
                    terremoto=terremoto,
                    pasta_terremotos=pasta_terremotos,
                    escala_terremoto=esc,
                    dt_fatorreducao=dt_fatorreducao,
                    num_pav=num_pav
                )

                # ── Análise Dinâmica ───────────────────────────────────────
                analise = analyze_earthquake2(
                    terremoto=terremoto,
                    pasta_terremotos=pasta_terremotos,
                    diretorio_analise=diretorio_analise,
                    T1_static=T1_static,
                    tolerancia=tolerancia,
                    dt_fatorreducao=dt_fatorreducao,
                    max_iter=max_iter,
                    cortar_sinal=cortar_sinal
                )

                # ── Modal Report 3: depois da análise dinâmica ─────────────
                if analise == 'SUCESSO':
                    analyze_modal(
                        numero_de_modos=numero_de_modos,
                        eps_amortecimento=eps_amortecimento,
                        salvar_figuras=False,
                        diretorio=None,
                        modal_filename=modal_dyn
                    )
                    print(f'  ModalReport 3 gerado (pós dinâmica): {modal_dyn}')

    # ── Pós-processamento ─────────────────────────────────────────────────
    if analise == 'SUCESSO':
        MaxIDR_pavs_x, MaxIDR, MaxDrift_Edificio = modelo2D_IDR_drift(
            terremoto=terremoto, Z_pavs=Z_pavs)
        MaxDESLOC = modelo2D_deslocamento(
            terremoto=terremoto, Z_pavs=Z_pavs)
        MaxVEL_ABS_pavs_x, MaxVEL_ABS, MaxVEL_Rel = modelo2D_velocidade(
            terremoto=terremoto, Z_pavs=Z_pavs,
            vel_history=vel_history, pasta_terremotos=pasta_terremotos)
        MaxAccels_ABS_pavs_x, MaxACCEL_ABS = modelo2D_aceleracaoABS(
            terremoto=terremoto, Z_pavs=Z_pavs)

        resultado_dic = modelo2D_resultados(
            X_spans=X_spans, Z_pavs=Z_pavs,
            b_col=b_col, h_col=h_col,
            Extremos_Inf_Discretiza_Col=Extremos_Inf_Discretiza_Col,
            Mid_discretiza_Col=Mid_discretiza_Col,
            Extremos_Sup_Discretiza_Col=Extremos_Sup_Discretiza_Col,
            b_beam=b_beam, h_beam=h_beam,
            Extremos_Discretiza_Beam=Extremos_Discretiza_Beam,
            Internos_Discretiza_Beam=Internos_Discretiza_Beam,
            Mid_discretiza_Beam=Mid_discretiza_Beam,
            analise_starttime=analise_starttime,
            terremoto=terremoto, analise=analise, escala_terremoto=esc,
            MaxIDR=MaxIDR, MaxIDR_pavs_x=MaxIDR_pavs_x,
            MaxDrift_Edificio=MaxDrift_Edificio,
            MaxDESLOC=MaxDESLOC, MaxVEL_ABS_pavs_x=MaxVEL_ABS_pavs_x,
            MaxVEL_ABS=MaxVEL_ABS, MaxVEL_Rel=MaxVEL_Rel,
            MaxAccels_ABS_pavs_x=MaxAccels_ABS_pavs_x, MaxACCEL_ABS=MaxACCEL_ABS,
            T1_static=T1_static, T1_sismo=T1_static,
            fc_normal=fc_normal, fy_normal=fy_normal_Column_TopBars,
            Es_normal=Es_normal, eps_amortecimento=eps_amortecimento,
            massPav=massPav,
            Sa_T1=Sa_T1, Sa_T_02s=Sa_T_02s, Sa_T_03s=Sa_T_03s,
            Sa_T_06s=Sa_T_06s, Sa_T_1s=Sa_T_1s,
            Sv_T1=Sv_T1, Sd_T1=Sd_T1,
            PGA=PGA, PGV=PGV, PGD=PGD,
            infill_wall=Infill_wall,
            resultados_tit=resultados_tit
        )

    else:
        # Sismo falhou — registra zeros para manter a linha no CSV
        zeros = [0.0] * (num_pav - 1)
        resultado = [
            terremoto, analise, esc, num_pav,
            0, zeros, 0, zeros, 0,
            0, 0, 0, 0, zeros, 0, zeros,
            0, zeros, 0, zeros,
            T1_static, T1_static,
            fc_normal, fy_normal_Column_TopBars, Es_normal,
            eps_amortecimento, massPav[num_pav - 1],
            Sa_T1, Sa_T_02s, Sa_T_03s, Sa_T_06s, Sa_T_1s,
            Sv_T1 * 9.80665 * 100, Sd_T1 * 9.80665 * 100,
            PGA, PGV * 100, PGD * 100,
            Infill_wall
        ]
        resultado_dic = {resultados_tit[i]: resultado[i]
                         for i in range(len(resultados_tit))}
        print(f"  *** SISMO {terremoto} FALHOU — zeros registrados ***")

    ResultadosFinais.append(resultado_dic)

    # ── Limpar recorders e resetar para o próximo sismo ──────────────────
    remover_analises()

    # ── Salvar CSV parcial (checkpoint) ──────────────────────────────────
    pd.DataFrame(ResultadosFinais).to_csv(f'./Results_Dynamic_Multisismos/{frame_folder}/Results_PARTIAL.out', index=False)
    print(f"  Checkpoint saved: Results_Dynamic_Multisismos/{frame_folder}/Results_PARTIAL.out  "
          f"({len(ResultadosFinais)}/{len(terremotos)} earthquakes)")


# =============================================================================
# RESULTADOS FINAIS CONSOLIDADOS
# =============================================================================

ResultadosFinais_df = pd.DataFrame(ResultadosFinais)
ResultadosFinais_df.to_csv(f'./Results_Dynamic_Multisismos/{frame_folder}/Results_ALL_EARTHQUAKES.out', index=False)
pickle.dump(ResultadosFinais, open(f'./Results_Dynamic_Multisismos/{frame_folder}/Results_ALL_EARTHQUAKES.pkl', 'wb'))

elapsed = time.time() - start_time
n_ok    = sum(1 for r in ResultadosFinais if r.get('Analysis') == 'SUCESSO')
n_fail  = len(ResultadosFinais) - n_ok

print(f"\n{'#'*60}")
print(f"  ANÁLISE CONCLUÍDA")
print(f"  Sismos analisados : {len(terremotos)}")
print(f"  SUCESSO           : {n_ok}")
print(f"  FALHOU            : {n_fail}")
print(f"  Tempo total       : {elapsed:.1f} s  ({elapsed/60:.1f} min)")
print(f"  Output file       : Results_Dynamic_Multisismos/{frame_folder}/Results_ALL_EARTHQUAKES.out")
print(f"{'#'*60}\n")
