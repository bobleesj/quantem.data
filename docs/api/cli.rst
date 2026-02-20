CLI
===

Command-line interface for ``quantem-data``.

.. code-block:: bash

   # list datasets
   quantem-data list
   quantem-data list --technique 4dstem

   # show metadata
   quantem-data info arina_32x32_48x48

   # list files on HF Hub
   quantem-data files

   # download
   quantem-data download arina_32x32_48x48

   # upload
   quantem-data upload my_data.npy --name silicon_110 --technique hrtem \
       --description "Silicon [110] HRTEM" --contributor "Jane Doe"

.. autofunction:: quantem.data.cli.main
