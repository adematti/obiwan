|

.. image:: _static/obiwan_logo.png
  :width: 500 px
  :align: center

|

.. title:: Obiwan docs

Foreword
========

This website is under construction. Previous version, by Kaylan Burleigh and John Moustakas is to be found here: `former Obiwan website <https://legacysurvey.github.io/obiwan/>`_.


.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/building
  user/data_model
  user/post_processing

.. toctree::
  :maxdepth: 1
  :caption: Developer documentation

  developer/docker
  developer/documentation
  developer/tests
  changes

.. toctree::
  :hidden:

Introduction
============

**Obiwan** is a Monte Carlo method for adding fake galaxies to `Legacy Survey <http://legacysurvey.org>`_ imaging data,
running the `legacypipe <https://github.com/legacysurvey/legacypipe>`_ pipeline, and repeating.


What for?
---------

Targets for spectroscopic follow-up are selected among the sources detected by **legacypipe**.
The target density includes cosmological clustering signal (to be measured) but is also impacted by so-called "imaging systematics",
due to the telescope, the opacity of the atmosphere, extinction and dust of the Milky Way, bias and variance of **legacypipe**.
These systematics can be (partly) removed by regressing the target density against photometric templates (linear model, or `neural nets <https://arxiv.org/abs/1907.11355v2>`_),
but these methods can only remove dependence of the target density on *known* systematics.

**Obiwan** rather forward models the source detection and target selection process, by injecting fake galaxies into raw images,
running the **legacypipe** and applying the target selection colour cuts.

**A picture is worth a 1000 words**

.. image:: _static/obiwan_1000_words.png
   :width: 800 px
   :align: center


Why the name *Obiwan*?
----------------------

Just as Obi-Wan Kenobi was the *only hope* in Star Wars: Episode IV - A New Hope (`YouTube <https://www.youtube.com/watch?v=0RDIJfoBhFU>`_); **Obiwan** is one of the only hopes for removing (most of) photometric systematics in the sample of galaxies selected from the imaging data.

Acknowledgements
----------------

See the `offical acknowledgements <http://legacysurvey.org/#Acknowledgements>`_ for the Legacy Survey.


Changelog
---------

* :doc:`changes`

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
