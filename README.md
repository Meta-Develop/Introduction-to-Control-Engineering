# Introduction to Control Engineering

<p align="right">
  <strong>English</strong> | <a href="docs/README.ja.md">日本語</a>
</p>

Introduction to Control Engineering is an open textbook for readers beginning the study of control systems. It starts from the mathematics of changing quantities and uses that language to build models, analyze feedback, and design controllers.

The Japanese manuscript is the source edition. Differential equations, transfer functions, frequency response, state-space methods, estimation, optimal control, model predictive control, and mechanics-based modeling are introduced through the main exposition rather than collected as a separate prerequisite list.

## Directory Layout

| Path | Purpose |
|------|---------|
| `ja/` | Japanese source edition |
| `en/` | English mirrored edition placeholder |
| `common/style/` | Shared LaTeX style |
| `docs/` | Curriculum maps and design notes |
| `pdf/` | Prebuilt PDFs for quick reading |

## Build

Prebuilt PDFs are available here:

- [Japanese PDF](pdf/control-engineering-ja.pdf)
- [English PDF](pdf/control-engineering-en.pdf)

To rebuild locally:

```sh
make ja
```

The English edition will mirror the Japanese edition after the Japanese section flow stabilizes.
