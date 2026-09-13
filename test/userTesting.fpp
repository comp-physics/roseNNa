program name
    #:def ranksuffix(RANK)
    $:'' if RANK == 0 else '(' + ':' + ',:' * (RANK - 1) + ')'
    #:enddef ranksuffix
    #:def genArray(arr)
    (/#{for index, x in enumerate(arr)}#${x}$#{if index < (len(arr)-1)}#, #{endif}##{endfor}#/)
    #:enddef genArray
    #:def gen(arr)
    #{for index, x in enumerate(arr)}#${x}$#{if index < (len(arr)-1)}#, #{endif}##{endfor}#
    #:enddef gen

    #:def rev(arr)
    [#{for x in range(len(arr),0,-1)}#${x}$ #{if x > 1}#, #{endif}##{endfor}#]
    #:enddef rev
    #:def revNum(num)
    [#{for x in range(num,0,-1)}#${x}$ #{if x > 1}#, #{endif}##{endfor}#]
    #:enddef revNum
    #:def genArrayNoParen(arr)
    #{for index, x in enumerate(arr)}#${x}$#{if index < (len(arr)-1)}#, #{endif}##{endfor}#
    #:enddef genArrayNoParen
    USE model
    USE reader
    implicit none
    #:mute
    #:include 'inputs.fpp'
    #:include 'variables.fpp'
    #:endmute
    #:for inp in inpShape
    REAL (c_double), DIMENSION(${genArrayNoParen(inpShape[inp])}$) :: ${inp}$
    #:endfor
    #:for o in outShape
    REAL (c_double), DIMENSION(${genArrayNoParen(o[1])}$) :: ${o[0]}$
    #:endfor
    #:for inp in arrs
    ${inp}$ = RESHAPE(${genArray(arrs[inp])}$,${genArray(inpShape[inp])}$, order = ${rev(inpShape[inp])}$)
    #:endfor

    CALL initialize()
    print *, "Model Reconstruction Success!"
    open(1, file = "test.txt")
    #:for inp in arrs
    ${inp}$ = RESHAPE(${genArray(arrs[inp])}$,${genArray(inpShape[inp])}$, order = ${rev(inpShape[inp])}$)
    #:endfor
    CALL use_model(#{for index,n in enumerate(inpShape)}#${n}$, #{endfor}##{for index,n in enumerate(outputs)}#${n}$#{if index < (len(outputs)-1)}#, #{endif}##{endfor}#)
    #:for x in outputs
    print *, ${x}$
    #:endfor
    #:for x in outShape
    #: set a = x[0]
    WRITE(1, *) SHAPE(${a}$)
    WRITE(1, *) PACK(RESHAPE(${x[0]}$,(/#{for num in range(len(x[1]),0,-1)}#SIZE(${x[0]}$, dim = ${num}$)#{if num > 1}#, #{endif}##{endfor}#/), order = [#{for x in range(len(x[1]),0,-1)}#${x}$#{if x > 1}#, #{endif}##{endfor}#]),.true.)
    #:endfor
end program name
