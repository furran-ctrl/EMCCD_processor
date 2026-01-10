import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import pandas as pd
#rom sympy import *
from scipy.interpolate import make_interp_spline

def plot_waterfall_PDF(Iat:np.array, data: list, scale: float, width_to_height_ratio: float, filename: str = 'waterfall_plot_final.png'):
    """
    Generates the final customized waterfall plot of Percentage Difference (PD).

    Incorporates fixed width, flipped XPS order, aligned baselines, and specific 
    y-axis range calculation.

    Args:
        data: list of [xps_value, [intensity values], [radial distance values]].
        xps_0: The reference XPS value for time calculation.
        pixel_to_q: Factor to convert radial distance to q (x-axis).
        scale: Vertical scaling factor. Used for PD labels' step size (e.g., 1.0 = 1% step).
        width_to_height_ratio: Desired ratio of the plot width to height.
        filename: Name for the saved plot image file.
    """
    if not data or len(data) < 2:
        print("Error: The data list is empty or has fewer than two groups.")
        return

    # 1. Sort data by XPS value and Reverse Order (Biggest XPS at the bottom, i=0)
    # Sort descending by XPS, then reverse to have smallest XPS (earliest time) at i=0
    sorted_data = sorted(data, key=lambda x: x[0])
    
    # 2. Identify and calculate Background (I_0) from the two largest XPS groups (now at the end)
    bg_group_1 = sorted_data[0] 
    bg_group_2 = sorted_data[1]
    I_0 = (np.array(bg_group_1[1]) + np.array(bg_group_2[1])) / 2
    xps_bg_1, xps_bg_2 = bg_group_1[0], bg_group_2[0]

    # 3. Setup Figure Dimensions (Width fixed at 8)
    plot_width = 8
    plot_height = plot_width / width_to_height_ratio
    
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    
    time_tick_positions = []
    
    # Define the constant vertical step in %PD for visual separation (e.g., 4% PD separation)
    vertical_step_PD = 1.0 * scale 

    colors = plt.cm.gist_earth(np.linspace(0.05, 0.8, 8))

    # 4. Process and Plot Each Group
    num_curves = len(sorted_data)
    for i, group in enumerate(sorted_data):
        xps_value = group[0]
        I = np.array(group[1])
        radial_distance = np.array(group[2])

        r = np.linspace(0.1,8,160)
        damp = 0.04
        q = radial_distance
        # Calculate PD
        with np.errstate(divide='ignore', invalid='ignore'):
            PDF = calculate_Pr_discrete(r, q, I, I_0, Iat, damp)
        
        # Vertical offset is proportional to the index 'i'
        vertical_offset = i * vertical_step_PD
        PD_curve = PDF + vertical_offset
        
        curve_color = colors[i % len(colors)]
        # Plot the curve
        ax.plot(q, PD_curve, color=curve_color, linewidth=1.2)
        
        # ALIGNMENT: Draw the horizontal baseline (PD=0) at the curve's offset position
        #ax.axhline(vertical_offset, color='gray', linestyle='--', linewidth=0.8, zorder=0)

        time_tick_positions.append(vertical_offset)

    
    # 5. Set Y-Axis Limits (to leave room for one extra offset on top and bottom)
    
    # The total vertical range of the plotted data is:
    # Max Y: (Max index * vertical_step_PD) + Max positive PD deviation
    max_data_y = time_tick_positions[-1] 
    # Min Y: (Min index * vertical_step_PD) + Min negative PD deviation
    min_data_y = time_tick_positions[0]  # Since time_tick_positions[0] is 0
    
    # Add one extra vertical step to the top and bottom limits
    y_limit_top = max_data_y + vertical_step_PD
    y_limit_bottom = min_data_y - vertical_step_PD
    
    ax.set_ylim(y_limit_bottom, y_limit_top)

    
    # 6. Set up the Right-Hand (Time) Axis
    ax_time = ax.twinx()
    ax_time.set_ylim(ax.get_ylim()) 

    # Calculate time
    time_labels = [f"{group[0]:.3f} ps" for group in sorted_data]
    ax_time.set_yticks(time_tick_positions)
    ax_time.set_yticklabels(time_labels, fontsize=10, ha='left', va='center') # va='center' ensures perfect alignment with baseline
    ax_time.tick_params(axis='y', length=0)
    
    
    # 7. Set up the Left-Hand (PD) Axis Ticks
    
    # The PD labels should be placed at the baseline of each curve (vertical_offset)
    # The labels should be 0%, (N-1)*scale %, (N-1)*scale*2 % ...
    
    # Calculate the label value for each baseline (offset)
    # PD Label at i=0 is 0%.
    # PD Label at i=1 is vertical_step_PD * scale % (if scale is 1, it's 4%)
    # PD Label at i=2 is vertical_step_PD * 2 * scale % (if scale is 1, it's 8%)
    
    # NOTE: Since the curves are already shifted by vertical_step_PD, 
    # the labels should reflect the CUMULATIVE SHIFT.
    
    # PD labels for each baseline: 0%, 4%, 8%, ...
    # The `vertical_offset` array already represents the baseline positions (0, 4, 8, ...)
    pd_baseline_labels = [f"{v:.2f}\\%" for v in time_tick_positions]

    ax.set_yticks(time_tick_positions)
    ax.set_yticklabels(pd_baseline_labels)
    ax.tick_params(axis='y', which='major', length=5) 
    
    ax.set_xticks(np.arange(0, 8, 1))

    # 8. Set Labels and Title
    title = (f"Waterfall Plot of Percentage Difference (PD) - Aligned\n"
             f"Background $I_0$ averaged from $XPS$ at ${xps_bg_1}$ and ${xps_bg_2}$ (Reversed Order)")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('$q$ (Normalized Radial Distance)', fontsize=12)
    ax.set_ylabel('Percentage Difference ($\%PD$)', fontsize=12, loc='top')
    
    ax.grid() # Turn off all default grid lines
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Final aligned waterfall plot saved as '{filename}'")

def calculate_Pr_discrete(r_array, s, I, I0, Iat, k):
    """
    基于离散采样点计算P(r)数组：
    P(r) = r * ∫[(I-I0)/Iat * s*sin(sr)*exp(-k s²)] ds
    
    参数说明：
    ----------
    r_array : np.array (任意维度)
        径向坐标r的数组，如 [0.1, 0.2, 0.3] 或 [[1,2],[3,4]]
    s : np.array (1维)
        采样点s的数组（需单调递增），长度N
    I : np.array (1维)
        s对应点的I值，长度N（与s等长）
    I0 : np.array (1维)
        s对应点的I0值，长度N（与s等长）
    Iat : np.array (1维)
        s对应点的Iat值，长度N（与s等长）
    k : float
        衰减参数k
    
    返回：
    -------
    P_r_array : np.array
        与r_array同维度的P(r)计算结果数组
    """
    # 1. 预处理：避免除零和数值异常
    Iat_safe = np.where(np.isclose(Iat, 0), np.finfo(float).eps, Iat)  # 替换0为极小值
    #term_base = (I - I0) / Iat_safe  # (I-I0)/Iat 项（长度N）
    term_base = (I) / Iat_safe
    
    # 2. 基础项计算（与r无关的部分，长度N）
    s_term = s  # s项
    exp_term = np.exp(-k * s**2)  # exp(-k s²) 项
    base = term_base * s_term * exp_term  # 长度N
    
    # 3. 扁平化r数组，方便向量化计算
    r_flat = r_array.ravel()  # 转为1维数组
    P_r_flat = np.zeros_like(r_flat, dtype=np.float64)
    
    # 4. 遍历每个r值计算积分（向量化计算被积项）
    for i, r in enumerate(r_flat):
        # 计算当前r对应的sin(sr)项（长度N）
        sin_term = np.sin(s * r)
        # 完整被积项（长度N）
        integrand = base * sin_term
        # 辛普森法积分（simpson自动处理区间，s需单调）
        integral = simpson(y=integrand, x=s)
        # 计算P(r)
        P_r_flat[i] = r * integral
    
    # 5. 恢复r数组的原始形状
    P_r_array = P_r_flat.reshape(r_array.shape)
    
    return term_base



path_dcs=r"C:\Users\ab177\Desktop\Cipher\3.7MeV\3.7MeV/"
table=pd.read_csv(path_dcs+'Periodic_Table.csv')
N=230

def no_to_sym(ele_no):
    return table['Symbol'][ele_no-1]

def read_dat_dcs(atom_no,path_dcs):
    atom_sym=no_to_sym(atom_no)
    path=path_dcs+atom_sym+'.dat'
    with open(path,'r') as file:
        a=file.read()
    a0=a.split('\n')
    data=np.empty(N)
    for i in range(N):
        a31=str(a0[31+i]).split(' ')
        data[i]=a31[6]
    return data**0.5

def import_DCS(max_at_no):
    f=np.empty((max_at_no+1,N))
    for i in range(max_at_no):
        f[i+1]=read_dat_dcs(i+1,path_dcs)
    return f

def electron_gamma(Ek):
    """
    计算电子的洛伦兹因子gamma和波数k（避免符号运算，纯数值计算）
    参数：
        Ek: 电子动能，单位eV
    返回：
        gamma: 洛伦兹因子（float）
        k: 波数（1/m，float）
    物理常数（国际单位制）：
        qe: 电子电荷 (C)
        me: 电子静质量 (kg)
        c: 真空中光速 (m/s)
        h: 普朗克常数 (J·s)
    """
    # 物理常数（国际单位制）
    qe = 1.602176565e-19   # 电子电荷 (C)
    me = 9.10938291e-31    # 电子静质量 (kg)
    c = 299792458          # 光速 (m/s)
    h = 6.62606957e-34     # 普朗克常数 (J·s)
    
    # 1. 直接计算洛伦兹因子gamma（核心优化：无需求解速度）
    rest_energy = me * c**2  # 电子静能 (J)
    Ek_joule = Ek * qe       # 动能转换为焦耳
    total_energy = rest_energy + Ek_joule  # 总能量
    gamma = total_energy / rest_energy     # 洛伦兹因子（解析解）
    
    # 2. 计算动量p (kg·m/s)
    # 推导：p = sqrt((E/c)^2 - (me c)^2) = me c * sqrt(gamma² - 1)
    p = me * c * np.sqrt(gamma**2 - 1)
    
    # 3. 计算德布罗意波长和波数k
    lamb = h / p            # 波长 (m)
    k = 2 * np.pi / lamb    # 波数 (1/m)
    
    return float(gamma), k

def import_s():
    qe=1.602176565e-19 
    me=9.10938291e-31
    c=299792458 
    h=6.62606957e-34
    E=3700000*qe+me*c**2 #kinetic energy=3.7MeV
    p=(E**2/c**2-me**2*c**2)**0.5
    lamb=h/p
    k=2*np.pi/lamb #wave vector of the incident electron

    path=path_dcs+'C.dat'
    with open(path,'r') as file:
        a=file.read()
    a0=a.split('\n')
    theta_deg=np.empty(N)
    for i in range(N):
        a31=str(a0[31+i]).split(' ')
        theta_deg[i]=a31[2]
    
    theta=theta_deg*np.pi/180
    S=2*k*np.sin(0.5*theta)
    s=np.array(S)
    return s

def import_DCS_interpolated(max_at_no,Ek,s):
    gamma,k0=electron_gamma(3.7*1e6)
    gamma1,k01=electron_gamma(Ek*1e6)
    f=import_DCS(max_at_no)*gamma1/gamma
    f=f*1e8/0.529177
    s0=import_s()*1e-10*0.529177
    f1=np.zeros((len(f),len(s)))
    for i in range(len(f)-1):
        i+=1
        f1[i]=make_interp_spline(s0,f[i])(s)
    return f1

#f=import_DCS_interpolated(9,3.7,radial_bins_converted)
#Iat = 2f[1]**2 + f[8]**2