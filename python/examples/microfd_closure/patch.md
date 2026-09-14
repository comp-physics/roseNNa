# patch.md: wiring the closure into microfd.c

This documents the edits to `microfd.c` that add the per-cell closure from
`closure.py` to the solver's viscous flux. It is documentation: nothing in
this repository compiles microfd, the diff is not applied anywhere, and no
microfd source is vendored here. Line numbers and surrounding context are
taken from the 231-line reference copy of `microfd.c` used to design this
patch. Device residency for the generated code is unvalidated until
`rosenna gpu-gate` has run on a GPU machine (see README.md); this patch is
correspondingly "compiles and runs on the host; device path unvalidated".

The closure model is `closure.onnx` (9 inputs: the velocity-gradient tensor
du_i/dx_j, flattened as du/dx, du/dy, du/dz, dv/dx, dv/dy, dv/dz, dw/dx,
dw/dy, dw/dz; 16 hidden units, Tanh; 1 output, no activation), generated
with:

```
rosenna generate closure.onnx --lang c --precision double --out .
```

into the same directory as `microfd.c`. The plan embeds (177 parameters),
so `closure_infer` is `static inline` in `closure.h` with the weights baked
in: no `closure_init`, and nothing to link for the per-point path below.
`closure_infer_batch` (used only by the batched alternative at the end of
this file) is not header-inline; it always lives in `closure.c` / the
kernel's `.cu` file, so that path links `libclosure.a`.

## 1. Include the header

```diff
 #include <mpi.h>
 #include <omp.h>
+#include "closure.h"                          // closure_infer: 9 velocity gradients -> nut, header-inline (embedded plan)
 
 #define NV 5                                   // rho, rho u, rho v, rho w, E
```

## 2. A place to keep the turbulent viscosity: `g.nut`

`g` already holds one padded-block array per quantity (`q`, `q1`, `w`, `F`,
the halo buffers); `nut` is one more, one value per cell (not `NV*nc` like
`q`/`w`, since it is a single scalar field).

```diff
   double L[3], o[3], h[3], gamma, mu, pr, cfl, tend, t;
-  double *q, *q1, *w, *F, *sbuf[2], *rbuf[2];      // F holds all three directions: [d][NV][nc]
+  double *q, *q1, *w, *F, *sbuf[2], *rbuf[2];      // F holds all three directions: [d][NV][nc]
+  double *nut;                                     // turbulent/SGS viscosity from the closure, one value per cell
   MPI_Comm comm;
```

## 3. The `closure()` kernel

Placed after `prim()` (which fills `w`, the primitives array `closure()`
reads) and before `face()` (which reads `g.nut`), so it slots directly into
`rhs_eval`'s existing sequence. It uses microfd's own `LOCALS`/`FOR3`/`IDX`
macros and its own naming convention for the primitives array (`w[nc+c]` =
u, `w[2*nc+c]` = v, `w[3*nc+c]` = the third velocity component, named `s`
throughout microfd.c to avoid colliding with the `w` array itself). The
gradients are ordinary second-order central differences at the cell center,
distinct from `face()`'s one-sided/averaged stencil at a face:

```diff
 static void prim(const double*q){                                        // conserved -> primitive over the whole padded block
   LOCALS; double*w=g.w; const double gm=g.gamma-1;
   #pragma omp target teams loop
   for(size_t c=0;c<nc;c++){ double r=q[c],u=q[nc+c]/r,v=q[2*nc+c]/r,s=q[3*nc+c]/r;
     w[c]=r; w[nc+c]=u; w[2*nc+c]=v; w[3*nc+c]=s; w[4*nc+c]=gm*(q[4*nc+c]-.5*r*(u*u+v*v+s*s)); }
 }
 
+static void closure(void){                     // per-cell turbulent viscosity from the velocity-gradient closure model
+  LOCALS; const double*w=g.w; double*nut=g.nut; const double h0=g.h[0],h1=g.h[1],h2=g.h[2];
+  FOR3(NG,NG,NG,){
+    const long c=IDX(i,j,k);
+    const double *u=w+nc+c, *v=w+2*nc+c, *s=w+3*nc+c;              // u, v, s: the three velocity components at this cell
+    double feat[9]={ (u[1]-u[-1])/(2*h0), (u[sx]-u[-sx])/(2*h1), (u[sy]-u[-sy])/(2*h2),
+                      (v[1]-v[-1])/(2*h0), (v[sx]-v[-sx])/(2*h1), (v[sy]-v[-sy])/(2*h2),
+                      (s[1]-s[-1])/(2*h0), (s[sx]-s[-sx])/(2*h1), (s[sy]-s[-sy])/(2*h2) };
+    closure_infer(feat, nut+c);                 // header-inline; declare-target already covers it, no extra decoration needed here
+  }
+}
+
 static void face(int d){                                               // flux through the face c+1/2 normal to d, stored in F at cell c
```

`FOR3(NG,NG,NG,)` expands to microfd's own
`#pragma omp target teams loop collapse(3)` over the interior cells, so
`closure_infer` runs inside the same offloaded loop nest as every other
kernel here -- it needs no pragma of its own, because `closure.h` already
wraps it in a guarded `omp declare target` region (rulings this plan
enforces on every generated header).

Caveat, stated plainly rather than glossed over: `FOR3(NG,NG,NG,...)` covers
only the interior physical cells, while `face()` reads `g.nut` one cell
outside that range on the low side of each direction (its own loop starts
at `NG-(d==0)` etc., to reach the boundary face using the adjacent ghost
cell). A production integration would need `nut` extended into that ghost
layer -- e.g. by including `closure()` in `halo()`'s boundary/exchange logic,
or by writing constant extrapolation there -- which this worked example does
not do, to keep the diff focused on the closure call itself.

## 4. Call it from `rhs_eval`

```diff
-static void rhs_eval(double*q){ halo(q); prim(q); for(int d=0;d<3;d++) face(d); }
+static void rhs_eval(double*q){ halo(q); prim(q); closure(); for(int d=0;d<3;d++) face(d); }
```

## 5. The `muf` line in `face`

Inside the existing `if(mu>0)` viscous block, blend the molecular viscosity
with the face-averaged turbulent viscosity from the two cells straddling
the face, and use that blend (`muf`, not `mu`) in the stress:

```diff
     if(mu>0){                                                            // viscous stress and heat flux at the face, 2nd-order central
       const long st[3]={1,sx,sy}; const double h[3]={h0,h1,h2}; double du[3][3], div=0;
+      const double muf=mu+.5*(g.nut[c]+g.nut[c+s]);                      // molecular + face-averaged closure viscosity
       for(int a=0;a<3;a++) for(int b=0;b<3;b++) if(a==b||a==d||b==d){ const double*u=w+(1+a)*nc+c; const long t=st[b];   // off-normal off-diagonal terms are dead
         du[a][b]= b==d ? (u[s]-u[0])/h[d] : (u[t]-u[-t]+u[s+t]-u[s-t])/(4*h[b]); }  // normal: two cells; tangential: averaged central
       for(int a=0;a<3;a++) div+=du[a][a];
       f[4]-=kap*(w[4*nc+c+s]/w[c+s]-w[4*nc+c]/w[c])/h[d];                              // heat flux with T = p/rho
-      for(int m=0;m<3;m++){ const int a=(d+m)%3; const double tau=mu*(du[a][d]+du[d][a]-(a==d)*2./3*div);
+      for(int m=0;m<3;m++){ const int a=(d+m)%3; const double tau=muf*(du[a][d]+du[d][a]-(a==d)*2./3*div);
         f[1+m]-=tau; f[4]-=tau*.5*(w[(1+a)*nc+c]+w[(1+a)*nc+c+s]); }
     }
```

`kap` (the heat-flux conductivity) is left on the molecular `mu`, computed
earlier in `face()` from `g.mu`; blending it too would need a turbulent
Prandtl number, which is outside the scope of this example.

## 6. Allocate and map `g.nut`

In `main`, alongside the other padded-block arrays:

```diff
   const size_t m=NV*g.nc; double**arr[]={&g.q,&g.q1,&g.w,&g.F,&g.sbuf[0],&g.sbuf[1],&g.rbuf[0],&g.rbuf[1]};
   for(int i=0;i<8;i++) if(!(*arr[i]=calloc(i<3?m:i==3?3*m:g.nbuf,sizeof(double)))) die("out of memory");
+  if(!(g.nut=calloc(g.nc,sizeof(double)))) die("out of memory");        // one turbulent-viscosity value per cell
   { LOCALS; for(int k=0;k<g.e[2];k++) for(int j=0;j<g.e[1];j++) for(int i=0;i<g.e[0];i++){   // IC on the padded block, ghosts included
```

and add it to the same `target enter data` that maps everything else,
right after the pointers it captures as locals:

```diff
-  double *q=g.q,*q1=g.q1,*w=g.w,*F=g.F,*s0=g.sbuf[0],*s1=g.sbuf[1],*r0=g.rbuf[0],*r1=g.rbuf[1]; const size_t nb=g.nbuf, m3=3*m;
-  #pragma omp target enter data map(to:q[0:m]) map(alloc:q1[0:m],w[0:m],F[0:m3],s0[0:nb],s1[0:nb],r0[0:nb],r1[0:nb])
+  double *q=g.q,*q1=g.q1,*w=g.w,*F=g.F,*s0=g.sbuf[0],*s1=g.sbuf[1],*r0=g.rbuf[0],*r1=g.rbuf[1],*nut=g.nut; const size_t nb=g.nbuf, m3=3*m;
+  #pragma omp target enter data map(to:q[0:m]) map(alloc:q1[0:m],w[0:m],F[0:m3],s0[0:nb],s1[0:nb],r0[0:nb],r1[0:nb],nut[0:g.nc])
```

`nut` is `alloc`, not `to`: `closure()` computes it fresh on the device
every `rhs_eval`, exactly like `w`.

## 7. Building

```
make cpu EXTRA="-I."      # host build; closure.h on the include path
make      EXTRA="-I."     # nvc target; same include, offloaded to the GPU
```

`EXTRA` is microfd's own hook for extra compiler flags; `-I.` is enough
when `closure.h`, `closure.c`, `closure_kernel.cu` and `closure.mk` sit
next to `microfd.c` as `rosenna generate` left them. The per-point path
above needs nothing else -- `closure_infer` is header-inline. If a future
revision of this example switches to the batched alternative below, add
`libclosure.a` (built by `closure.mk`) to the link line as well.

## Alternative: a larger model, batched via `closure_infer_batch`

`closure_infer` is called once per cell above, which is appropriate for a
network this small (177 parameters: the call is dominated by the loop and
memory-access overhead microfd already pays for `prim`/`face`, not by the
MLP itself). A larger closure network changes that trade-off: gather every
cell's 9 features into one device array first, then make ONE call to the
native batched kernel instead of one `closure_infer` call per cell.

```c
static void closure_batched(void){
  LOCALS; const double *w=g.w; double *nut=g.nut; const double h0=g.h[0],h1=g.h[1],h2=g.h[2];
  static double *feat=0;                        // [nc][9], allocated and mapped once
  if(!feat){
    feat=malloc(sizeof(double)*9*nc);
    #pragma omp target enter data map(alloc:feat[0:9*nc])
  }
  FOR3(NG,NG,NG,){
    const long c=IDX(i,j,k); const double *u=w+nc+c,*v=w+2*nc+c,*s=w+3*nc+c;
    double *f9=feat+9*c;
    f9[0]=(u[1]-u[-1])/(2*h0); f9[1]=(u[sx]-u[-sx])/(2*h1); f9[2]=(u[sy]-u[-sy])/(2*h2);
    f9[3]=(v[1]-v[-1])/(2*h0); f9[4]=(v[sx]-v[-sx])/(2*h1); f9[5]=(v[sy]-v[-sy])/(2*h2);
    f9[6]=(s[1]-s[-1])/(2*h0); f9[7]=(s[sx]-s[-sx])/(2*h1); f9[8]=(s[sy]-s[-sy])/(2*h2);
  }
  int status;
  #pragma omp target data use_device_ptr(feat, nut)
  { status = closure_infer_batch((int)nc, feat, nut, 0); }   // ruling R5: feat, nut already on the device; no transfer here
  if(status) die("closure_infer_batch failed");
}
```

`closure()` becomes `closure_batched()` in the `rhs_eval` edit of section
4. Unlike `closure_infer`, `closure_infer_batch` is not header-inline, so
this alternative links `libclosure.a` (`closure.mk`, whichever
`ROSENNA_BACKEND` it was built with -- `cuda`, `hip` or `omp`) into
microfd's own build, in addition to `-I.`. The `feat` gather loop is a
separate `FOR3` pass over the same cells `closure()` covered above, so the
same ghost-layer caveat from section 3 applies here too.
