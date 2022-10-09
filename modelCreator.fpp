module model
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    ! USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    USE model_layers
    USE readTester
    ! ===============================================================

    
    IMPLICIT NONE
    contains
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
        #:mute
        #:include 'variables.fpp'
        
        #:def ranksuffix(RANK)
        $:'' if RANK == 0 else '(' + ':' + ',:' * (RANK - 1) + ')'
        #:enddef ranksuffix
        #:def genArray(arr)
        (/#{for index, x in enumerate(arr)}#${x}$#{if index < (len(arr)-1)}#, #{endif}##{endfor}#/)
        #:enddef genArray

        #:def genArrayNoParen(arr)
        #{for index, x in enumerate(arr)}#${x}$#{if index < (len(arr)-1)}#, #{endif}##{endfor}#
        #:enddef genArrayNoParen

        #:def genInput(arr)
        #{for index,n in enumerate(arr)}#i${index}$#{if index < (len(arr)-1)}#, #{endif}##{endfor}#
        #:enddef genInput

        #:def genOutput(arr)
        #{for index,n in enumerate(arr)}#o${index}$#{if index < (len(arr)-1)}#, #{endif}##{endfor}#
        #:enddef genOutput
        #:endmute
        
        SUBROUTINE use_model(${genInput(trueInputs)}$, ${genOutput(outShape)}$)
            IMPLICIT NONE
            !INPUTS CORRESPONDING TO C
            #:for index, inp in enumerate(trueInputs)
            REAL (c_double), INTENT(INOUT), DIMENSION(${genArrayNoParen(inp[1])}$) :: i${index}$
            #:endfor
            !===========================

            !INPUTS CORRESPONDING TO INTERMEDIARY PROCESSING
            #:for inp in inputs
            REAL (c_double), ALLOCATABLE, #{if any(inp[0] in sublist for sublist in trueInputs)}#INTENT(INOUT),#{endif}# DIMENSION${ranksuffix(inp[1])}$ :: ${inp[0]}$
            #:endfor
            !================================================

            !OUTPUTS CORRESPONDING TO C
            #:for index,o in enumerate(outShape)
            REAL (c_double), INTENT(OUT), DIMENSION(${genArrayNoParen(o[1])}$) :: o${index}$
            #:endfor
            !===========================

            #:for inp in trueInputs
            REAL (c_double), ALLOCATABLE, DIMENSION${ranksuffix(len(inp[1]))}$ :: ${inp[0]}$
            #:endfor

            REAL :: T1, T2

            #:for inp in trueInputs
            ALLOCATE(${inp[0]}$(${genArrayNoParen(inp[1])}$))
            #:endfor

            #:for index, inp in enumerate(trueInputs)
            ${inp[0]}$ = i${index}$
            #:endfor

            
            CALL CPU_TIME(T1)
            
            #: set layer_dict = {}
            #: for tup in architecture
            #: mute
            #: if tup[0] not in layer_dict
            $: layer_dict.update([(tup[0],1)])
            #: endif
            #: endmute
            #!Linear Layer
            #: if tup[0] == 'Gemm'
            !========Gemm Layer============
            CALL linear_layer(${tup[1][0]}$, linLayers(${layer_dict[tup[0]]}$),${1-tup[1][1]}$)
            #!LSTM Layer
            #: elif tup[0] == 'LSTM'
            !========LSTM Layer============
            CALL lstm(${tup[1][0]}$, ${tup[1][1]}$, ${tup[1][2]}$, lstmLayers(${layer_dict[tup[0]]}$)%whh, lstmLayers(${layer_dict[tup[0]]}$)%wih, lstmLayers(${layer_dict[tup[0]]}$)%bih, lstmLayers(${layer_dict[tup[0]]}$)%bhh, ${tup[2][0]}$)

            #!Convolutional Layer
            #: elif tup[0] == 'Conv'
            !========Conv Layer============
            CALL conv(${tup[1][0]}$, convLayers(${layer_dict[tup[0]]}$)%weights, convLayers(${layer_dict[tup[0]]}$)%biases, ${genArray(tup[2][0])}$, ${genArray(tup[2][2])}$, ${genArray(tup[2][3])}$)
            #!Max Pooling Layer
            #: elif tup[0] == 'MaxPool'
            !========MaxPool Layer============
            CALL max_pool(${tup[1][0]}$,maxpoolLayers(${layer_dict[tup[0]]}$), ${tup[2][0]}$, ${genArray(tup[2][1])}$, ${genArray(tup[2][2])}$)

            #!Average Pooling Layer
            #: elif tup[0] == 'AveragePool'
            !========MaxPool Layer============
            CALL avgpool(${tup[1][0]}$,avgpoolLayers(${layer_dict[tup[0]]}$), ${tup[2][0]}$, ${genArray(tup[2][1])}$, ${genArray(tup[2][2])}$)

            #!Transpose
            #: elif tup[0] == 'Transpose'
            !========Transpose============
            ${tup[1][0]}$ = RESHAPE(${tup[1][0]}$,(/#{for index, num in enumerate(tup[2][0])}#SIZE(${tup[1][0]}$, dim = ${num}$)#{if index < (len(tup[2][0])-1)}#, #{endif}##{endfor}#/), order = ${tup[2][0]}$)

            #!Reshape
            #: elif tup[0] == 'Reshape'
            !========Reshape============
            #: if tup[-1][0] == 0
            ${tup[1][0]}$ = RESHAPE(${tup[1][0]}$,(/#{for num in range(tup[1][1],0,-1)}#SIZE(${tup[1][0]}$, dim = ${num}$)#{if num > 1}#, #{endif}##{endfor}#/), order = [#{for x in range(tup[1][1],0,-1)}#${x}$#{if x > 1}#, #{endif}##{endfor}#])
            ${tup[2][0]}$ = RESHAPE(${tup[1][0]}$,(/#{for index, num in enumerate(tup[3][0])}#${num}$#{if index < (len(tup[3][0])-1)}#, #{endif}##{endfor}#/), order = [#{for x in range(len(tup[3][0]),0,-1)}#${x}$#{if x > 1}#, #{endif}##{endfor}#])
            #: else
            reshapeLayers(${layer_dict[tup[0]]}$)%reshape${tup[1][1]}$d = RESHAPE(reshapeLayers(${layer_dict[tup[0]]}$)%reshape${tup[1][1]}$d,(/#{for num in range(tup[1][1],0,-1)}#SIZE(reshapeLayers(${layer_dict[tup[0]]}$)%reshape${tup[1][1]}$d, dim = ${num}$)#{if num > 1}#, #{endif}##{endfor}#/), order = [#{for x in range(tup[1][1],0,-1)}#${x}$#{if x > 1}#, #{endif}##{endfor}#])
            ${tup[2][0]}$ = RESHAPE(reshapeLayers(${layer_dict[tup[0]]}$)%reshape${tup[1][1]}$d,(/#{for index, num in enumerate(tup[3][0])}#${num}$#{if index < (len(tup[3][0])-1)}#, #{endif}##{endfor}#/), order = [#{for x in range(len(tup[3][0]),0,-1)}#${x}$#{if x > 1}#, #{endif}##{endfor}#])
            #: endif

            #!Squeeze
            #: elif tup[0] == 'Squeeze'
            !========Squeeze============
            ${tup[2][0]}$ = RESHAPE(${tup[1][0]}$,(/#{for num in range(tup[1][1])}##{if num not in tup[3][0]}#SIZE(${tup[1][0]}$, dim = ${num+1}$)#{if num < (tup[1][1]-1)}#, #{endif}##{endif}##{endfor}#/))

            #!Add
            #: elif tup[0] == 'Add'
            !===========Add============
            #: if len(tup[2][1]) == 0
            ${tup[1][0]}$ = ${tup[1][0]}$ + RESHAPE(addLayers(${layer_dict[tup[0]]}$)%adder, ${genArray(tup[2][0][-tup[2][2]:])}$)
            #: else
            ${tup[1][0]}$ = ${tup[1][0]}$ + RESHAPE(broadc(addLayers(${layer_dict[tup[0]]}$)%adder,${genArray(tup[2][0])}$,RESHAPE(${genArray(tup[2][1])}$,${genArray([int(len(tup[2][1])/2),2])}$, order=[2,1])), ${genArray(tup[2][0][-tup[2][2]:])}$)
            #: endif

            #!MatMul
            #: elif tup[0] == 'MatMul'
            !=======MatMul=========
            CALL matmul${tup[2][0]}$D(${tup[1][0]}$, ${tup[1][1]}$)
            
            #!ReLu
            #: elif tup[0] == 'Relu'
            ${tup[1][0]}$ = relu${tup[2][0]}$d(${tup[1][0]}$)
            
            #!Tanh
            #: elif tup[0] == 'Tanh'
            ${tup[1][0]}$ = tanhh${tup[2][0]}$d(${tup[1][0]}$)

            #!Sigmoid
            #: elif tup[0] == 'Sigmoid'
            ${tup[1][0]}$ = sigmoid${tup[2][0]}$d(${tup[1][0]}$)

            #: endif
            #: mute
            $: layer_dict.update([(tup[0],layer_dict[tup[0]]+1)])
            #: endmute
            #: endfor
            call CPU_TIME(T2)
            #:for index,out in enumerate(outputs)
            o${index}$ = ${outputs[out]}$
            #:endfor
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================
