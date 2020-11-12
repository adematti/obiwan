Building the documentation
##########################

The documentation can be built from the Docker image, e.g. bind mount your directory ``obiwanabsolutepath``::

  docker run -v obiwanabsolutepath:/src/obiwan -it adematti/obiwan:dr9.3

Then::

  cd obiwan/doc
  make html

Changes to the ``rst`` files can be made from outside the Docker container.
You can display the website (outside the Docker container) by opening ``_build/html/index.html/``.

Finally, to push the documentation, `Read the Docs <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/read-the-docs.html>`_.
