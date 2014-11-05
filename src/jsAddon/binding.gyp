{
  "targets": [
    {
      "target_name": "addon",
      'actions': [
        {
          'action_name': 'build ann cuda part',
          'inputs': ['../ml/ann.cu'],
          'outputs': ['/home/estamm/Gits/curoy/src/ml/ann.o'],
          'action': ['make', '-C../ml'],
          'process_outputs_as_sources': 1
        },
        {
          'action_name': 'build library cuda part',
          'inputs': ['../lib/cuMatrix.cu'],
          'outputs': ['/home/estamm/Gits/curoy/src/lib/cudaMatrix.o'],
          'action': ['make', '-C../lib', '-B'],
          'process_outputs_as_sources': 1
        },
        {
          'action_name': 'build library cuda part',
          'inputs': ['../wt/WaveletTransformator.cpp'],
          'outputs': ['/home/estamm/Gits/curoy/src/wt/WaveletTransformator.o','/home/estamm/Gits/curoy/src/wt/Filter.o','/home/estamm/Gits/curoy/src/wt/SymmetricPadding.o'],
          'action': ['make', '-C../wt', '-B'],
          'process_outputs_as_sources': 1
        }
      ],
      'library_dirs': ['/usr/local/cuda-6.5/lib64','-l/usr/local/lib'],
      "sources": [ "addon.cpp", "AnnWrapper.cpp"],
      'libraries': ['-lcudart', '-lcuda', '-lcurand','-lnanodbc'],
      "cflags": ['-std=c++11', '-I/usr/local/cuda/include'],
      "include_dirs": [
        "<!(node -e \"require('nan')\")"
      ]
    }
  ]
}
