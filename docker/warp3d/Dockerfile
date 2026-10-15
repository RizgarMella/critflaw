# syntax=docker/dockerfile:1
#
# WARP3D feasibility spike -- OpenMP (threads) build with Intel ifx + MKL (Pardiso)
#
#   docker build -t critflaw-warp3d:spike .
#
# Stage 1 installs only the Intel Fortran compiler and the MKL "core devel"
# apt packages (not the full oneAPI HPC kit), builds warp3d.exe with WARP3D's
# own src/Makefile.linux, and collects the Intel runtime shared libraries the
# executable needs. Stage 2 is a plain Ubuntu image with the executable, those
# runtime libraries, and the example / verification problems.
#
############################  stage 1: build  ################################
FROM ubuntu:24.04 AS builder

ARG WARP3D_REPO=https://github.com/rhdodds/warp3d
ARG WARP3D_COMMIT=13f53d31270a6b27bb313985bb90f30e3d1fcd45
ARG ONEAPI_VER=2025.3
ARG JOBS=8
# WARP3D's Makefile.linux default is
#   -axAVX,CORE-AVX2,COMMON-AVX512 -xIVYBRIDGE -align array64byte -qopt-zmm-usage=high
# The -x<code> options add an Intel-CPU check to the main program; this
# workstation has an AMD CPU, so use the vendor-neutral -march form instead.
ARG MACHINE_TYPE="-march=core-avx2 -align array64byte"

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git make gcc g++ libc6-dev binutils time \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
      | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
      > /etc/apt/sources.list.d/oneAPI.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      intel-oneapi-compiler-fortran-${ONEAPI_VER} \
      intel-oneapi-mkl-core-devel-${ONEAPI_VER} \
 && rm -rf /var/lib/apt/lists/* \
 && dpkg -l | grep intel-oneapi > /intel_packages.txt

# WARP3D source at a pinned commit (shallow fetch of exactly that commit)
ENV WARP3D_HOME=/opt/warp3d
RUN git init -q ${WARP3D_HOME} && cd ${WARP3D_HOME} \
 && git remote add origin ${WARP3D_REPO} \
 && git fetch -q --depth 1 origin ${WARP3D_COMMIT} \
 && git checkout -q FETCH_HEAD \
 && git log -1 --format='%H %cI %s' > ${WARP3D_HOME}/SPIKE_COMMIT.txt \
 && rm -rf .git

# Build. Makefile.linux expects static MKL + libiomp5.so in linux_packages/lib
# (WARP3D normally downloads these from Dropbox via install_libs_from_remote;
# here they come from the Intel apt packages instead).
RUN source /opt/intel/oneapi/setvars.sh > /dev/null \
 && cd ${WARP3D_HOME} \
 && mkdir -p linux_packages/lib run_linux obj_linux spike_build \
 && ln -s ${MKLROOT}/lib/libmkl_intel_lp64.a ${MKLROOT}/lib/libmkl_intel_thread.a \
          ${MKLROOT}/lib/libmkl_core.a linux_packages/lib/ \
 && cp -L ${CMPLR_ROOT}/lib/libiomp5.so linux_packages/lib/ \
 && cd src \
 && (ifx --version | tee ${WARP3D_HOME}/spike_build/ifx_version.txt) \
 && echo "MKLROOT=${MKLROOT}" | tee -a ${WARP3D_HOME}/spike_build/ifx_version.txt \
 && touch main_program.f \
 && start=$(date +%s) \
 && (make -j ${JOBS} -f Makefile.linux MACHINE_TYPE="${MACHINE_TYPE}" 2>&1 \
       | tee ${WARP3D_HOME}/spike_build/make.log | tail -40) \
 && end=$(date +%s) \
 && echo "make_wall_seconds=$((end-start)) jobs=${JOBS} machine_type=${MACHINE_TYPE}" \
       | tee ${WARP3D_HOME}/spike_build/make_time.txt \
 && test -x ../run_linux/warp3d.exe \
 && ls -l ../run_linux/warp3d.exe | tee -a ${WARP3D_HOME}/spike_build/make_time.txt

# Fact-finding: does a program built with WARP3D's default -xIVYBRIDGE run on this CPU?
RUN source /opt/intel/oneapi/setvars.sh > /dev/null \
 && mkdir /tmp/x && cd /tmp/x \
 && printf '      program t\n      print *, "xflag test ran OK"\n      end\n' > t.f \
 && ( grep -m1 'model name' /proc/cpuinfo; \
      ifx -xIVYBRIDGE t.f -o t_x 2>&1; ./t_x 2>&1; echo "exit status (-xIVYBRIDGE): $?"; \
      ifx -axAVX,CORE-AVX2,COMMON-AVX512 -xIVYBRIDGE t.f -o t_ax 2>&1; ./t_ax 2>&1; \
      echo "exit status (WARP3D default flags): $?" ) \
      | tee ${WARP3D_HOME}/spike_build/xflag_test.txt ; true

# Collect the Intel runtime shared libraries the executable actually needs and
# the Intel licence / redistribution texts; drop build-only material.
RUN source /opt/intel/oneapi/setvars.sh > /dev/null \
 && cd ${WARP3D_HOME} \
 && rm -f linux_packages/lib/*.a \
 && ldd run_linux/warp3d.exe | tee spike_build/ldd_build.txt \
 && for f in $(ldd run_linux/warp3d.exe | awk '$3 ~ /^\/opt\/intel/ {print $3}'); do \
        cp -Lv "$f" linux_packages/lib/ ; done \
 && mkdir -p spike_build/intel_licensing \
 && cp -rL /opt/intel/oneapi/licensing spike_build/intel_licensing/ 2>/dev/null ; \
    for f in $(find /opt/intel/oneapi -iname 'redist*.txt' -o -iname 'third-party-programs*.txt' | head -40); do \
        d=spike_build/intel_licensing/$(echo "$f" | sed 's#/opt/intel/oneapi/##; s#/#__#g'); cp -L "$f" "$d"; done ; \
    cp /intel_packages.txt spike_build/ \
 && rm -rf obj_linux old_code manual_tex_files RVE_support tecplot \
 && ls -l linux_packages/lib

############################  stage 2: runtime  ##############################
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 time \
 && rm -rf /var/lib/apt/lists/*

ENV WARP3D_HOME=/opt/warp3d
COPY --from=builder /opt/warp3d /opt/warp3d
ENV LD_LIBRARY_PATH=/opt/warp3d/linux_packages/lib

# `warp3d <threads> < input > output`  (same behaviour as warp3d_script_linux)
RUN printf '%s\n' '#!/bin/bash' \
      'export OMP_NUM_THREADS=${1:-1}' \
      'export MKL_NUM_THREADS=$OMP_NUM_THREADS' \
      'ulimit -s 100000 2>/dev/null || ulimit -s unlimited 2>/dev/null' \
      'exec "$WARP3D_HOME/run_linux/warp3d.exe"' > /usr/local/bin/warp3d \
 && chmod +x /usr/local/bin/warp3d \
 && ldd /opt/warp3d/run_linux/warp3d.exe | tee /opt/warp3d/spike_build/ldd_runtime.txt \
 && ! grep -q 'not found' /opt/warp3d/spike_build/ldd_runtime.txt

WORKDIR /work
CMD ["bash"]
