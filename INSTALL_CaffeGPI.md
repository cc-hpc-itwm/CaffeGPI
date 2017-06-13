# Installation of the CaffeGPI

1)
Prepare and install your machine as you do for installing the standard Caffe.
Compile the standard Caffe to check if all dependies are resolved on your
machine.

2)
Go to http://www.gpi-site.com/gpi2. Download the current version of GPI-2
and install as described there. Use the Infiniband network if availabe
using the --with-infiniband switch for installation. Otherwise use the
--with-ethernet option.

3)
Clone the CaffeGPI repository.

4)
Build the CaffeGPI executable using Cmake:

cd <top level directory of CaffeGPI>
mkdir build
cd build
CMAKE_INCLUDE_PATH=<paths> CMAKE_PREFIX_PATH=<GPI-2 path> cmake-gui ..

CMAKE_PREFIX_PATH is used to indicate the location of your GPI-2 installation,
e.g. CMAKE_PREFIX_PATH=/opt/GPI-2-1.3.0. If cmake-gui is not installed on your
machine, use ccmake instead.

Configure as usual. Uncheck the BUILD_SHARED_LIBS box to use static linking.

In the cmake-gui press the button "Configure". The paths of the GPI-2 library and
the include directory should be found automatically. If not, specify them
manually in the GPI2 part of the gui. If you installed GPI-2 with Infiniband
support, the path to the ibverbs library is needed additionally. Specify this
path if it is not detected automatically.

After "Configure" finished without errors, press the button "Generate" in the
cmake-gui. Exit the gui.

Compile CaffeGPI as usual typing "make". The executable is created in
tools/caffe.

5)
Create a machine file:

Define your (multi node) machine. Edit a text file containing one line
per Caffe process. Each line contains the hostname of the computer which should
be used to run the process. If more than one process on the same
compute node is needed, all these should form a continous block. If you use
GPUs for computing then use one process per GPU. If you calculate "CPU only"
then use one process per NUMA node on your computers. Example: Let's say you
have two cluster nodes (node46 and node47) with 2 GPUs each, your machine file
would probably look like this:

node46
node46
node47
node47

6)
Make sure that all the files accessed during your calculatation are accessible
from all the nodes of your machine file with the same absolute path.  

7)
Make sure you can login to all the nodes of your machine file with ssh
and without providing a password. Create e.g. a keypair using "ssh-keygen"
and distribute the keys to the remote nodes.

8)
Modify your nets and solvers:

a)
Change all the paths in your net and solver files to absolute paths.

b)
Replace all your data_param layers with "parallel_data_param" layers
in your net.

c)
Reduce the batch sizes in all the data_param layers according to the size
of your machine file. If a final batch size of 256 is intended and your
machine runs on 4 CaffeGPI processes, then provide a batch size of 64 in
the specification of your net.

9)
Start CaffeGPI:

Login to the first node of your machine file.
Call CaffeGPI typing:

<GPI-2 path>/bin/gaspi_run -m <path>/machine.txt <Caffe path>/caffe <Caffe options>

Use full paths for all the files specified here. The file machine.txt
is your machine file.

If you have more than one NUMA node on your compute node and you have one
CaffeGPI process per NUMA node (e.g. if you compute "CPU only"), then use the
-N switch of gaspi_run to increase performance.
