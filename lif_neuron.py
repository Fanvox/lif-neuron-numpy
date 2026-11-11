"""漏积分放电（Leaky Integrate-and-Fire, LIF）神经元的纯 NumPy 实现。

模型（单位：mV, nA, MΩ, ms）
    tau_m * dV/dt = -(V - E_L) + R_m * I
    V >= V_th 时：在 t_spike 发放一个脉冲，并令 V = V_reset
    绝对不应期内 V 固定在 V_reset

运行方式：
    python lif_neuron.py
        完整演示：打印频率对照表，生成三张教学图。

    python lif_neuron.py demo --no-plots
        只打印结果，不绘图（不需要 matplotlib）。

    python lif_neuron.py run --current 3.0 --tau 10 --v-threshold -50
        自定义一组参数，生成单条膜电位轨迹。

依赖：numpy（仿真）、matplotlib（仅绘图）。
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# 1. 参数与结果的数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LIFParams:
    """LIF 的全部参数。默认值对应一个典型的皮层兴奋性神经元。"""

    tau_m: float = 20.0          # 膜时间常数 (ms)
    e_leak: float = -70.0        # 漏电反转电位 E_L (mV)
    v_threshold: float = -55.0   # 阈值 V_th (mV)
    v_reset: float = -70.0       # 脉冲后的重置电位 V_reset (mV)
    r_m: float = 10.0            # 膜电阻 R_m (MΩ)
    t_refractory: float = 2.0    # 绝对不应期 (ms)

    def __post_init__(self) -> None:
        if self.tau_m <= 0 or self.r_m <= 0:
            raise ValueError("tau_m 和 r_m 必须为正数")
        if self.v_reset >= self.v_threshold:
            raise ValueError("必须满足 v_reset < v_threshold")
        if self.e_leak >= self.v_threshold:
            raise ValueError("必须满足 e_leak < v_threshold")
        if self.t_refractory < 0:
            raise ValueError("不应期不能为负")

    @property
    def rheobase_na(self) -> float:
        """使稳态膜电位恰好等于阈值的最小恒定电流 (nA)。"""
        return (self.v_threshold - self.e_leak) / self.r_m

    @property
    def steady_state_mv(self) -> float:
        """零输入时膜电位的稳态值（就是 E_L）。"""
        return self.e_leak


@dataclass
class LIFResult:
    """一次仿真的全部输出。"""

    t_ms: np.ndarray             # 时间轴 (ms)
    v_mv: np.ndarray             # 膜电位 (mV)，脉冲时刻记录为 V_th
    spikes_ms: np.ndarray        # 脉冲时刻 (ms)
    spike_indices: np.ndarray    # 脉冲对应的数组下标（方便画图）
    current_na: float
    params: LIFParams


# ---------------------------------------------------------------------------
# 2. 核心：仿真一个 LIF 神经元
# ---------------------------------------------------------------------------
def simulate_lif(
    params: LIFParams,
    current_na: float,
    duration_ms: float = 300.0,
    dt_ms: float = 0.05,
) -> LIFResult:
    """在恒定输入电流下仿真一个 LIF 神经元。

    这里没有用显式欧拉法，而是在每个时间步内使用指数解的精确形式：

        V(t+dt) = V_inf + (V(t) - V_inf) * exp(-dt / tau_m)
        V_inf   = E_L + R_m * I

    这样即使 dt 取得稍大，也不会像欧拉法那样“冲过头”或数值爆炸。
    """
    if duration_ms <= 0 or dt_ms <= 0:
        raise ValueError("duration_ms 和 dt_ms 必须为正数")

    n_steps = int(round(duration_ms / dt_ms))
    if n_steps < 1:
        raise ValueError("duration_ms 至少应包含一个 dt_ms 时间步")
    t = np.arange(n_steps, dtype=float) * dt_ms
    v = np.full(n_steps, params.e_leak, dtype=float)

    v_inf = params.e_leak + params.r_m * current_na
    decay = np.exp(-dt_ms / params.tau_m)

    spikes: list[float] = []
    spike_indices: list[int] = []

    v_state = params.e_leak   # 当前时刻真实的膜电位
    refractory_until = -np.inf

    for k, t_now in enumerate(t):
        # 绝对不应期：把膜电位钳在 V_reset，不积分
        if t_now < refractory_until - 1e-12:
            v[k] = params.v_reset
            continue

        # 两个脉冲之间：用指数解更新一步
        v_next = v_inf + (v_state - v_inf) * decay

        if v_next >= params.v_threshold:
            # 记录脉冲。为了让图上看得到“到达阈值”，该采样点保留 V_th；
            # 真正的重置从下一个采样点开始生效。
            v[k] = params.v_threshold
            spikes.append(t_now)
            spike_indices.append(k)

            v_state = params.v_reset
            refractory_until = t_now + params.t_refractory
        else:
            v[k] = v_next
            v_state = v_next

    return LIFResult(
        t_ms=t,
        v_mv=v,
        spikes_ms=np.asarray(spikes, dtype=float),
        spike_indices=np.asarray(spike_indices, dtype=int),
        current_na=float(current_na),
        params=params,
    )


# ---------------------------------------------------------------------------
# 3. 从脉冲序列计算放电频率
# ---------------------------------------------------------------------------
def firing_rate_hz(result: LIFResult, discard_ms: float = 200.0) -> float:
    """用“脉冲间隔的中位数”估计稳态放电频率。

    开头一段包含从静息电位第一次爬到阈值的瞬态，所以丢掉前 discard_ms。
    使用中位数而不是平均值，可以减小个别采样抖动的影响。
    """
    spikes = result.spikes_ms[result.spikes_ms >= discard_ms]
    if spikes.size < 2:
        return 0.0
    median_isi = float(np.median(np.diff(spikes)))
    return 1000.0 / median_isi


def theoretical_fi(current_na: np.ndarray | float, params: LIFParams) -> np.ndarray:
    """恒定电流下 f-I 曲线的解析解（单位 Hz）。

    当 I <= I_rh 时没有稳态放电。当 I > I_rh 时，阈值之间的时间间隔为

        T_isi = t_ref + tau_m * ln((V_inf - V_reset) / (V_inf - V_th))

    其中 V_inf = E_L + R_m * I。因此 f = 1000 / T_isi。
    """
    currents = np.atleast_1d(np.asarray(current_na, dtype=float))
    v_inf = params.e_leak + params.r_m * currents
    above = v_inf > params.v_threshold

    rates = np.zeros_like(currents)
    isi = np.full_like(currents, np.inf)
    isi[above] = params.t_refractory + params.tau_m * np.log(
        (v_inf[above] - params.v_reset)
        / (v_inf[above] - params.v_threshold)
    )
    rates[above] = 1000.0 / isi[above]
    return rates


# ---------------------------------------------------------------------------
# 4. 绘图（matplotlib 只在真正画图时导入，方便无绘图环境下复用仿真）
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_time_series(
    results: list[LIFResult],
    filename: str | Path = "figures/lif_timeseries.png",
) -> None:
    """画出不同输入强度下的膜电位和脉冲时序。"""
    plt = _setup_matplotlib()

    fig, axes = plt.subplots(
        len(results), 1,
        figsize=(10, 3.2 if len(results) == 1 else 2.4 * len(results)),
        sharex=True, constrained_layout=True,
    )
    if len(results) == 1:
        axes = [axes]

    colors = ["#2563eb", "#dc2626", "#059669"]
    for ax, result, color in zip(axes, results, colors):
        p = result.params
        ax.plot(result.t_ms, result.v_mv, color=color, lw=1.4, label="膜电位 V(t)")
        ax.axhline(p.v_threshold, color="#6b7280", ls="--", lw=1.0,
                   label=f"阈值 V_th = {p.v_threshold:g} mV")
        if result.spikes_ms.size:
            ax.vlines(result.spikes_ms, p.v_threshold, 5.0,
                      color="#ef4444", lw=0.9, alpha=0.8)
            ax.plot(result.spikes_ms, np.full(result.spikes_ms.shape, 5.0),
                    "v", color="#ef4444", ms=5, label="脉冲")
        rate = firing_rate_hz(result, discard_ms=50.0)
        ax.set_ylabel("V (mV)")
        ax.set_ylim(-80, 15)
        ax.set_title(
            f"输入 I = {result.current_na:g} nA   "
            f"（f ≈ {rate:.1f} Hz）",
            loc="left", fontsize=11,
        )
        ax.grid(alpha=0.25)

    from matplotlib.lines import Line2D

    p = results[0].params
    handles = [
        Line2D([0], [0], color="#2563eb", lw=1.5, label="膜电位 V(t)"),
        Line2D([0], [0], color="#6b7280", ls="--", lw=1.0,
               label=f"阈值 V_th = {p.v_threshold:g} mV"),
        Line2D([0], [0], color="#ef4444", marker="v", ls="none",
               ms=6, label="脉冲"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.995), ncol=3,
               frameon=False, fontsize=9)
    axes[-1].set_xlabel("时间 t (ms)")
    axes[-1].set_xlim(0, results[0].t_ms[-1])
    fig.suptitle("LIF 神经元：恒定输入下的积分—发放—重置",
                 x=0.02, ha="left", fontsize=14)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    print(f"已保存：{filename}")


def plot_fi_curve(
    filename: str | Path = "figures/lif_fi_curve.png",
) -> None:
    """扫描输入电流，画 f-I 曲线，并检查数值结果与解析解是否一致。"""
    plt = _setup_matplotlib()

    base = LIFParams()
    current_grid = np.linspace(0.0, 5.0, 800)
    test_currents = np.linspace(1.55, 5.0, 12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   constrained_layout=True)

    # ---- 左图：只改变膜时间常数 tau_m ----
    tau_values = [10.0, 20.0, 40.0]
    tau_colors = ["#2563eb", "#d97706", "#059669"]
    for tau, color in zip(tau_values, tau_colors):
        p = LIFParams(tau_m=tau)
        theory = theoretical_fi(current_grid, p)
        ax1.plot(current_grid, theory, color=color, lw=2.0,
                 label=f"tau_m = {tau:g} ms")
        currents = test_currents[test_currents > p.rheobase_na]
        sim_rates = [
            firing_rate_hz(simulate_lif(p, i, duration_ms=1200.0))
            for i in currents
        ]
        ax1.plot(currents, sim_rates, "o", color=color, ms=4.5,
                 mec="white", mew=0.8)
    ax1.axvline(base.rheobase_na, color="#9ca3af", ls=":", lw=1.0)
    ax1.set_xlabel("输入电流 I (nA)")
    ax1.set_ylabel("放电频率 f (Hz)")
    ax1.set_title("改变膜时间常数 tau_m", loc="left")
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 200)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", fontsize=9)

    # ---- 右图：只改变阈值 V_th ----
    vth_values = [-55.0, -50.0, -45.0]
    vth_colors = ["#2563eb", "#d97706", "#dc2626"]
    for vth, color in zip(vth_values, vth_colors):
        p = LIFParams(v_threshold=vth)
        theory = theoretical_fi(current_grid, p)
        ax2.plot(current_grid, theory, color=color, lw=2.0,
                 label=f"V_th = {vth:g} mV")
        currents = test_currents[test_currents > p.rheobase_na]
        sim_rates = [
            firing_rate_hz(simulate_lif(p, i, duration_ms=1200.0))
            for i in currents
        ]
        ax2.plot(currents, sim_rates, "o", color=color, ms=4.5,
                 mec="white", mew=0.8)
    for vth, color in zip(vth_values, vth_colors):
        p = LIFParams(v_threshold=vth)
        ax2.axvline(p.rheobase_na, color=color, ls=":", lw=1.0, alpha=0.6)
    ax2.set_xlabel("输入电流 I (nA)")
    ax2.set_ylabel("放电频率 f (Hz)")
    ax2.set_title("改变阈值 V_th", loc="left")
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 120)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "输入强度 vs 放电频率：实线 = 解析解，圆点 = NumPy 数值仿真"
        "（tau_m=20 ms, V_th=-55 mV, R_m=10 MΩ, t_ref=2 ms）",
        fontsize=10,
    )
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    print(f"已保存：{filename}")


def plot_mechanism(
    filename: str | Path = "figures/lif_mechanism.png",
) -> None:
    """放大一个放电周期，标注“积分、达阈、重置、不应期”四个阶段。"""
    plt = _setup_matplotlib()

    p = LIFParams()
    result = simulate_lif(p, current_na=2.5, duration_ms=100.0)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)

    ax.plot(result.t_ms, result.v_mv, color="#2563eb", lw=2.0,
            label="膜电位 V(t)")
    ax.axhline(p.v_threshold, color="#6b7280", ls="--", lw=1.0,
               label=f"阈值 V_th = {p.v_threshold:g} mV")
    ax.axhline(p.v_reset, color="#9ca3af", ls=":", lw=1.0,
               label=f"重置 V_reset = {p.v_reset:g} mV")

    # 用红色竖线表示脉冲；LIF 不建模动作电位的真实形状
    for i, t_spike in enumerate(result.spikes_ms):
        ax.vlines(t_spike, p.v_threshold, 22.0, color="#ef4444", lw=1.4,
                  label="脉冲" if i == 0 else None)
        ax.plot([t_spike], [22.0], "v", color="#ef4444", ms=7)
        ax.axvspan(t_spike, t_spike + p.t_refractory,
                   color="#f59e0b", alpha=0.18, lw=0)

    # 挑第二个和第三个脉冲标注，避免文字挤在左边
    if result.spikes_ms.size >= 4:
        t1, t2 = result.spikes_ms[1], result.spikes_ms[2]
        ax.annotate("① 积分：向 V_inf 指数上升",
                    xy=(t1 + 0.45 * (t2 - t1), -62.0),
                    xytext=(24, -40), fontsize=10,
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1))
        ax.annotate("② 越过阈值，发放脉冲",
                    xy=(t2, p.v_threshold),
                    xytext=(64, -35), fontsize=10, color="#dc2626",
                    arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1))
        ax.annotate("③ 重置到 V_reset",
                    xy=(t2 + 0.2, p.v_reset),
                    xytext=(62, -77), fontsize=10,
                    arrowprops=dict(arrowstyle="->", color="#374151", lw=1))
        ax.text(t1 + 0.5 * p.t_refractory, -75, "④ 不应期",
                fontsize=9, color="#b45309", ha="center", va="center")

    ax.set_xlim(0, 100)
    ax.set_ylim(-80, 30)
    ax.set_xlabel("时间 t (ms)")
    ax.set_ylabel("膜电位 V (mV)")
    ax.set_title("LIF 的积分—发放—重置循环（I = 2.5 nA）",
                 loc="left", pad=32)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.14),
              ncol=4, fontsize=8, frameon=False)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=180)
    plt.close(fig)
    print(f"已保存：{filename}")


# ---------------------------------------------------------------------------
# 5. 命令行入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lif_neuron.py",
        description="LIF 神经元教学示例：仿真、调参、画 f-I 曲线。",
    )
    parser.add_argument("--version", action="version",
                        version=f"lif-neuron {__version__}")
    subparsers = parser.add_subparsers(dest="command",
                                       metavar="{demo,run}")

    demo = subparsers.add_parser(
        "demo", help="生成对照表和全部三张教学图（默认）")
    demo.add_argument("--outdir", type=Path, default=Path("figures"),
                      help="图片输出目录（默认：figures）")
    demo.add_argument("--no-plots", action="store_true",
                      help="只打印结果，不绘图")

    run = subparsers.add_parser(
        "run", help="自定义参数，仿真一条膜电位轨迹")
    run.add_argument("--current", type=float, default=2.0,
                     help="恒定输入电流，单位 nA（默认：2.0）")
    run.add_argument("--tau", type=float, default=20.0,
                     help="膜时间常数，单位 ms（默认：20）")
    run.add_argument("--v-threshold", type=float, default=-55.0,
                     help="阈值，单位 mV（默认：-55）")
    run.add_argument("--v-reset", type=float, default=-70.0,
                     help="重置电位，单位 mV（默认：-70）")
    run.add_argument("--t-ref", type=float, default=2.0,
                     help="绝对不应期，单位 ms（默认：2）")
    run.add_argument("--r-m", type=float, default=10.0,
                     help="膜电阻，单位 MΩ（默认：10）")
    run.add_argument("--duration", type=float, default=500.0,
                     help="仿真时长，单位 ms（默认：500）")
    run.add_argument("--dt", type=float, default=0.05,
                     help="时间步长，单位 ms（默认：0.05）")
    run.add_argument("--outdir", type=Path, default=Path("figures"),
                     help="图片输出目录（默认：figures）")
    run.add_argument("--no-plots", action="store_true",
                     help="只打印结果，不绘图")
    return parser


def _matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


def _warn_missing_matplotlib() -> None:
    print("\n未安装 matplotlib，已跳过绘图；仿真结果仍然有效。")
    print("安装后即可出图：pip install matplotlib")


def _run_demo(args: argparse.Namespace) -> None:
    params = LIFParams()
    print("模型参数：")
    print(f"  tau_m = {params.tau_m:g} ms, E_L = {params.e_leak:g} mV, "
          f"V_th = {params.v_threshold:g} mV, V_reset = {params.v_reset:g} mV")
    print(f"  R_m = {params.r_m:g} MΩ, t_ref = {params.t_refractory:g} ms")
    print(f"  基强度 I_rh = {params.rheobase_na:.2f} nA\n")

    print("不同输入电流下的放电频率：")
    print("  I (nA)   脉冲数   f_sim (Hz)   f_theory (Hz)")
    for current in [1.2, 2.0, 2.5, 3.5]:
        result = simulate_lif(params, current, duration_ms=500.0)
        f_sim = firing_rate_hz(result, discard_ms=100.0)
        f_theory = float(theoretical_fi(current, params)[0])
        print(f"  {current:5.1f}   {result.spikes_ms.size:5d}   "
              f"{f_sim:9.2f}   {f_theory:11.2f}")

    if not args.no_plots:
        if not _matplotlib_available():
            _warn_missing_matplotlib()
        else:
            plot_time_series(
                [simulate_lif(params, i, duration_ms=200.0)
                 for i in (1.2, 2.0, 3.5)],
                args.outdir / "lif_timeseries.png",
            )
            plot_fi_curve(args.outdir / "lif_fi_curve.png")
            plot_mechanism(args.outdir / "lif_mechanism.png")

    print("\n手动调参：")
    print("  python lif_neuron.py run --current 3.0 --tau 10 "
          "--v-threshold -50")
    print("  或：python lif_neuron.py demo --no-plots")


def _run_custom(args: argparse.Namespace) -> None:
    params = LIFParams(
        tau_m=args.tau,
        v_threshold=args.v_threshold,
        v_reset=args.v_reset,
        t_refractory=args.t_ref,
        r_m=args.r_m,
    )
    result = simulate_lif(
        params, args.current,
        duration_ms=args.duration, dt_ms=args.dt,
    )
    discard = min(200.0, 0.2 * args.duration)
    rate = firing_rate_hz(result, discard_ms=discard)
    theory = float(theoretical_fi(args.current, params)[0])

    print("仿真参数：")
    print(f"  I = {args.current:g} nA, tau_m = {params.tau_m:g} ms, "
          f"V_th = {params.v_threshold:g} mV, V_reset = {params.v_reset:g} mV")
    print(f"  R_m = {params.r_m:g} MΩ, t_ref = {params.t_refractory:g} ms, "
          f"dt = {args.dt:g} ms, T = {args.duration:g} ms")
    print(f"  基强度 I_rh = {params.rheobase_na:.2f} nA")
    print(f"  脉冲数 = {result.spikes_ms.size}, "
          f"f_sim = {rate:.2f} Hz, f_theory = {theory:.2f} Hz")
    if result.spikes_ms.size:
        preview = ", ".join(f"{s:.2f}" for s in result.spikes_ms[:8])
        suffix = " ..." if result.spikes_ms.size > 8 else ""
        print(f"  前几个脉冲时刻 (ms)：{preview}{suffix}")
    else:
        print("  没有脉冲：膜电位未越过阈值。")

    if not args.no_plots:
        if not _matplotlib_available():
            _warn_missing_matplotlib()
        else:
            plot_time_series(
                [result],
                args.outdir / "custom_trace.png",
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        # 不带子命令时等价于 `python lif_neuron.py demo`
        args = parser.parse_args(["demo"])
    if args.command == "demo":
        _run_demo(args)
    else:
        try:
            _run_custom(args)
        except ValueError as exc:
            parser.error(str(exc))
    return 0


if __name__ == "__main__":
    main()
