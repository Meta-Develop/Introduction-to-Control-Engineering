"""Shared readability cleanup for generated textbook figures."""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import warnings

from matplotlib.figure import Figure


AXIS_LABEL_TRANSLATIONS = {
    r"時間 $t$ [s]": r"Time $t$ [s]",
    "時刻 t [s]": "Time t [s]",
    r"時刻 $t$ [s]": r"Time $t$ [s]",
    "時刻 [s]": "Time [s]",
    r"正規化時間 $\xi=t/\tau$ [-]": r"Normalized time $\xi=t/\tau$ [-]",
    r"正規化応答 $x=\theta/\theta_0$ [-]": r"Normalized response $x=\theta/\theta_0$ [-]",
    r"正規化出力 $y/(K u_0)$ [-]": r"Normalized output $y/(K u_0)$ [-]",
    r"温度偏差 $\theta$ [K]": r"Temperature deviation $\theta$ [K]",
    "位置 q [m]": "Position q [m]",
    "累積量": "Accumulated quantity",
    r"変化率 $r(t)$": r"Rate $r(t)$",
    "状態 x": "State x",
    r"状態 $x_1$": r"State $x_1$",
    r"状態 $x_2$": r"State $x_2$",
    r"状態 $x_1$ [-]": r"State $x_1$ [-]",
    r"状態 $x_2$ [-]": r"State $x_2$ [-]",
    r"状態 $x_k$ [-]": r"State $x_k$ [-]",
    r"状態 $x(t)$ [-]": r"State $x(t)$ [-]",
    r"出力 $f(x)$": r"Output $f(x)$",
    r"出力 $y(t)$ [-]": r"Output $y(t)$ [-]",
    r"$x$ 成分": r"$x$ component",
    r"$y$ 成分": r"$y$ component",
    r"入力偏差 $\tilde{v}=v-v_e$ [V]": r"Input deviation $\tilde{v}=v-v_e$ [V]",
    r"ヒータ入力寄与 $\Delta\dot{T}_{u}$ [K/s]": r"Heater input contribution $\Delta\dot{T}_{u}$ [K/s]",
    r"偏差 $z(t)$ [V]": r"Deviation $z(t)$ [V]",
    r"偏差 $e(t)$ [-]": r"Error $e(t)$ [-]",
    r"水位 $h(t)$ [m]": r"Water level $h(t)$ [m]",
    r"角度 $\theta(t)$ [rad]": r"Angle $\theta(t)$ [rad]",
    r"角度 $\theta$ [rad]": r"Angle $\theta$ [rad]",
    r"角速度 $\omega$ [rad/s]": r"Angular velocity $\omega$ [rad/s]",
    "第1成分 [-]": "Component 1 [-]",
    "第2成分 [-]": "Component 2 [-]",
    r"実部 $\Re L(j\omega)$ [-]": r"Real part $\Re L(j\omega)$ [-]",
    r"虚部 $\Im L(j\omega)$ [-]": r"Imaginary part $\Im L(j\omega)$ [-]",
    r"実部 $\sigma$ [1/s]": r"Real part $\sigma$ [1/s]",
    r"虚部 [rad/s]": r"Imaginary part [rad/s]",
    r"角周波数 $\omega$ [rad/s]": r"Angular frequency $\omega$ [rad/s]",
    "ゲイン [dB]": "Gain [dB]",
    "位相 [deg]": "Phase [deg]",
    r"位相 $\theta$ [rad]": r"Phase $\theta$ [rad]",
    r"決定変数 $u_0$ [-]": r"Decision variable $u_0$ [-]",
    r"決定変数 $u_1$ [-]": r"Decision variable $u_1$ [-]",
    r"入力 $u_k$ [-]": r"Input $u_k$ [-]",
    r"入力 $u$ [-]": r"Input $u$ [-]",
    r"入力指令 $u_c$ [-]": r"Commanded input $u_c$ [-]",
    r"入力・積分状態 [-]": r"Input / integrator state [-]",
    r"測定値 $y$ [-]": r"Measurement $y$ [-]",
    r"位置偏差 $x_1$ [-]": r"Position deviation $x_1$ [-]",
    r"位置誤差 $e_1$ [-]": r"Position error $e_1$ [-]",
    r"速度誤差 $e_2$ [-/s]": r"Velocity error $e_2$ [-/s]",
    r"変位 $q(t)$ [m]": r"Displacement $q(t)$ [m]",
    r"力 $u(t)$ [N]": r"Force input $u(t)$ [N]",
    r"終端偏差 $|r-q(T)|$ [m]": r"Terminal error $|r-q(T)|$ [m]",
    r"RMS 操作力 [N]": "RMS control force [N]",
    "終端からの後退ステップ数": "Backward steps from terminal",
    r"サンプル $k$ [-]": r"Sample $k$ [-]",
}


def translate_axis_label(label: str) -> str:
    """Translate common quantitative axis labels while preserving math notation."""
    return AXIS_LABEL_TRANSLATIONS.get(label, label)


def _clear_titles(fig: Figure) -> None:
    for text in list(fig.texts):
        text.set_text("")
        text.set_visible(False)

    for axis in fig.axes:
        for loc in ("left", "center", "right"):
            axis.set_title("", loc=loc)
        for title in (axis.title, getattr(axis, "_left_title", None), getattr(axis, "_right_title", None)):
            if title is not None:
                title.set_text("")
                title.set_visible(False)


def _raise_font_size(text, minimum: float) -> None:
    try:
        current = float(text.get_fontsize())
    except (TypeError, ValueError):
        current = minimum
    text.set_fontsize(max(current, minimum))


def _is_axis_or_legend_text(text, fig: Figure) -> bool:
    for axis in fig.axes:
        axis_texts = [
            axis.xaxis.label,
            axis.yaxis.label,
            axis.title,
            getattr(axis, "_left_title", None),
            getattr(axis, "_right_title", None),
        ]
        if hasattr(axis, "zaxis"):
            axis_texts.append(axis.zaxis.label)

        if any(text is axis_text for axis_text in axis_texts if axis_text is not None):
            return True
        if text in axis.get_xticklabels() or text in axis.get_yticklabels():
            return True
        if hasattr(axis, "zaxis") and text in axis.get_zticklabels():
            return True

        legend = axis.get_legend()
        if legend is not None:
            if text in legend.get_texts() or text is legend.get_title():
                return True

    return False


def _make_text_readable(fig: Figure) -> None:
    for axis in fig.axes:
        axis.set_xlabel(translate_axis_label(axis.get_xlabel()))
        axis.set_ylabel(translate_axis_label(axis.get_ylabel()))
        if hasattr(axis, "set_zlabel"):
            axis.set_zlabel(translate_axis_label(axis.get_zlabel()))

        axis.xaxis.label.set_size(max(float(axis.xaxis.label.get_size()), 12.0))
        axis.yaxis.label.set_size(max(float(axis.yaxis.label.get_size()), 12.0))
        if hasattr(axis, "zaxis"):
            axis.zaxis.label.set_size(max(float(axis.zaxis.label.get_size()), 12.0))

        axis.tick_params(axis="both", which="major", labelsize=10)
        axis.tick_params(axis="both", which="minor", labelsize=10)
        if hasattr(axis, "zaxis"):
            axis.tick_params(axis="z", which="major", labelsize=10)
            axis.tick_params(axis="z", which="minor", labelsize=10)

        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                _raise_font_size(text, 10.0)
            if legend.get_title() is not None:
                _raise_font_size(legend.get_title(), 10.0)

    for text in fig.findobj(match=lambda item: hasattr(item, "get_text") and hasattr(item, "get_fontsize")):
        if text.get_visible() and text.get_text() and not _is_axis_or_legend_text(text, fig):
            _raise_font_size(text, 8.0)


def finalize_figure(
    fig: Figure,
    output_path: str | Path,
    *,
    layout: Literal["tight", "none"] = "tight",
) -> None:
    """Apply the project figure-readability contract and save without resizing."""
    _clear_titles(fig)
    _make_text_readable(fig)
    if layout == "tight":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.995), pad=0.85, h_pad=1.0, w_pad=1.0)
    fig.savefig(output_path, facecolor="white")
