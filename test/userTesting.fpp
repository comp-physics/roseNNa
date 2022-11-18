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
    REAL :: T1, T2
    REAL (c_double), DIMENSION(100) :: times
    integer :: time
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
    ! open(56,file='graphs/timesF.txt',action='write',position='append') !delete this line
    ! DO time=1,100 !delete this line
    #:for inp in arrs
    ${inp}$ = RESHAPE(${genArray(arrs[inp])}$,${genArray(inpShape[inp])}$, order = ${rev(inpShape[inp])}$)
    #:endfor
    CALL CPU_TIME(T1)
    CALL use_model(#{for index,n in enumerate(inpShape)}#${n}$, #{endfor}##{for index,n in enumerate(outputs)}#${n}$#{if index < (len(outputs)-1)}#, #{endif}##{endfor}#)
    CALL CPU_TIME(T2)
    times(time) = T2-T1
    ! END DO !delete this line
    ! CALL bubble_sort(times) !delete this line
    ! print *, "Time taken: ", times(size(times,1)/2) !delete this line
    ! WRITE(56, *) times(size(times,1)/2), " " !delete this line
    #:for x in outputs
    print *, ${x}$
    #:endfor
    #:for x in outShape
    #: set a = x[0]
    WRITE(1, *) SHAPE(${a}$)
    WRITE(1, *) PACK(RESHAPE(${x[0]}$,(/#{for num in range(len(x[1]),0,-1)}#SIZE(${x[0]}$, dim = ${num}$)#{if num > 1}#, #{endif}##{endfor}#/), order = [#{for x in range(len(x[1]),0,-1)}#${x}$#{if x > 1}#, #{endif}##{endfor}#]),.true.)
    #:endfor
    contains
        subroutine bubble_sort(array)
            implicit none
            REAL (c_double), intent(inout) :: array(:)
            real :: temp
            integer :: i,j,last

            last=size(array)
            do i=last-1,1,-1
            do j=1,i
                if (array(j+1).lt.array(j)) then
                    temp=array(j+1)
                    array(j+1)=array(j)
                    array(j)=temp
                endif
            enddo
            enddo

        end subroutine bubble_sort
end program name
