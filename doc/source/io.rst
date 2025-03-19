===================
I/O Functionalities
===================

The :doc:`Dataset <functionalities>` class can parse several fMRI file types:

Functional Files
----------------

- Supported formats: `*.mat`, `*.nii.gz`, `*.gii`
- Outputs a **2D DataFrame** indexed by subject/run/task IDs.
- Compatible with `BIDS <https://bids.neuroimaging.io/index.html>`_ format.

.. image:: imgs/df_func.png
   :align: center
   :alt: Functional File Example

Experiment Files
----------------

Designed for `exptools2 <https://github.com/VU-Cog-Sci/exptools2>`_, this package can extract experiment phases.

.. image:: imgs/df_onsets.png
   :align: center
   :alt: Experiment File Example

Eyetracker Files
----------------

Supports `*.edf`-files from `EyeLink <https://www.sr-research.com/eyelink-1000-plus/>`_, using `hedfpy <https://github.com/tknapen/hedfpy>`_.
