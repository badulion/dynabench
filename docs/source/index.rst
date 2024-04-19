:html_theme.sidebar_secondary.remove: true

***********************
DynaBench documentation
***********************

**Date**: |today| **Version**: |release|

**Download documentation**: https://docs.scipy.org/doc/

**Useful links**:
`Install <start>`__ |
`Source Repository <https://github.com/badulion/dynabench>`__ |
`Dataset <https://wuedata.uni-wuerzburg.de/radar/de/dataset/sSEeRraAYDgQCgBP>`__ |

**DynaBench** (pronounced "Die-Nah-Bench") is an open-source benchmark dataset for evaluating deep learning models for all sort of task concerning physical spatiotemporal systems.

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :img-top: _static/book-solid.svg
        :text-align: center

        **Getting Started**
        ^^^

        The Getting Started guide will help you get started with DynaBench. 
        It shows how to download the data use our data iterator.

        +++

        .. button-ref:: guide
            :color: secondary
            :click-parent:

            To the guide

    .. grid-item-card::
        :img-top: _static/wrench-solid.svg
        :text-align: center

        **API reference**
        ^^^

        The reference guide contains a description of the DynaBench components. 

        +++

        .. button-ref:: api
            :color: secondary
            :click-parent:

            To the API reference

    .. grid-item-card::
        :img-top: _static/chart-line-solid.svg
        :text-align: center

        **Benchmark results**
        ^^^

        The benchmark results show the performance of different models on the DynaBench dataset. 

        +++

        .. button-ref:: results
            :color: secondary
            :click-parent:

            To the results

    .. grid-item-card::
        :img-top: _static/file-lines-solid.svg
        :text-align: center

        **Paper**
        ^^^

        The DynaBench paper has been published in the Joint European Conference on Machine Learning and Knowledge Discovery in Databases 2023.
        The paper describes the benchmark and the results in detail.

        +++

        .. button-link:: https://arxiv.org/abs/2306.05805
            :color: secondary
            :click-parent:

            To the paper


.. toctree::
    :maxdepth: 1
    :hidden:

    About <about>
    Getting Started <start>
    User Guide <guide>
    Benchmark Results <results>
    API reference <api>