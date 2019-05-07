TEST="0"
DATA="/data-2/gunrock_dataset/large"
DATA2="/data-2/topc-datasets"

for file in ak2010 belgium_osm coAuthorsDBLP delaunay_n10 delaunay_n13 delaunay_n21 webbase-1M soc-LiveJournal1 kron_g500-logn21
do
  if [ "$TEST" = "0" ] ; then
    echo bin/gpr --timing 1 --mxvmode 0 --niter 5 --max_niter 10 $DATA/$file/$file.mtx
    bin/gpr --timing 1 --mxvmode 0 --niter 5 --max_niter 10 $DATA/$file/$file.mtx
  fi
done

for file in soc-orkut soc-LiveJournal1 hollywood-2009 indochina-2004 rmat_n22_e64 rmat_n23_e32 rmat_n24_e16 rgg_n24_0.000548 road_usa
do
  if [ "$TEST" = "1" ] ; then
    echo bin/gpr --timing 1 --mxvmode 0 --niter 5 --max_niter 10 $DATA2/$file/$file.mtx
    bin/gpr --timing 1 --mxvmode 0 --niter 5 --max_niter 10 $DATA2/$file/$file.mtx
  fi
done

for file in test_bc test_cc test_mesh test_mis test_pr small chesapeake
do
  if [ "$TEST" = "2" ] ; then
    echo bin/gpr --timing 1 --mxvmode 0 --niter 5 --max_niter 10 data/small/$file.mtx
    bin/gpr --timing 1 --mxvmode 0 --niter 5 --max_niter 10 data/small/$file.mtx
  fi
done
