dask-ssh \
--nprocs 5 \
--nthreads 1 \
$(dig +short pinac-a.cs.kuleuven.be) \ 
$(dig +short pinac-b.cs.kuleuven.be) \ 
$(dig +short pinac-c.cs.kuleuven.be) \ 
$(dig +short pinac-d.cs.kuleuven.be) \ 
$(dig +short pinac30.cs.kuleuven.be) \ 
$(dig +short pinac23.cs.kuleuven.be) \ 
$(dig +short pinac24.cs.kuleuven.be)
