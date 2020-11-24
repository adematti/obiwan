Documentation
=============

Please follow `Sphinx style guide`_ when writing the documentation (except for filenames).

Building
--------

The documentation can be built from the Docker image, e.g. bind mount your directory ``obiwanabsolutepath``::

  docker run -v obiwanabsolutepath:/homedir/obiwan -it {dockerimage}

Then::

  cd $HOME/obiwan/doc
  make html

Changes to the ``rst`` files can be made from outside the Docker container.
You can display the website (outside the Docker container) by opening ``_build/html/index.html/``.

Finally, to push the documentation, `Read the Docs`_.


References
----------

.. target-notes::

.. _`Sphinx style guide`: https://documentation-style-guide-sphinx.readthedocs.io/en/latest/style-guide.html

.. _`Read the Docs`: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/read-the-docs.html
