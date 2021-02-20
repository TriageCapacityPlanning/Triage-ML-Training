from setuptools import setup, find_packages

setup(name='triage_ml',
      version='0.1',
      description='Triage ML model repository and training tool',
      url='https://github.com/TriageCapacityPlanning/Triage-ML-Training',
      packages=find_packages(),
      install_requires=['pytest', 'tensorflow-gpu', 'psycopg2-binary', 'numpy', 'matplotlib', 'requests'],
      entry_points={
          'console_scripts': [
                'triage-train=triage_ml.train:main',
                'triage-gendata=triage_ml.data.gen_data:main'
          ]
      },
      zip_safe=False)
