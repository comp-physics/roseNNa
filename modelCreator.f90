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

<<<<<<< HEAD
        SUBROUTINE use_model(input, output)
            IMPLICIT NONE
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: input
            REAL, ALLOCATABLE, INTENT(OUT), DIMENSION(:,:,:,:) :: output
            REAL :: T1, T2
            CALL CPU_TIME(T1)
            
            !========MaxPool Layer============
            CALL max_pool(input,maxpoolLayers(1), 0,         (/0, 0, 0, 0/),         (/3, 3/))

            call CPU_TIME(T2)

            output = input

=======
        SUBROUTINE use_model(Input3, Plus214_Output_0)
            IMPLICIT NONE
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: Input3
            REAL, ALLOCATABLE,  DIMENSION(:,:,:,:) :: Parameter5
            REAL, ALLOCATABLE,  DIMENSION(:,:,:) :: Parameter6
            REAL, ALLOCATABLE,  DIMENSION(:,:,:,:) :: Parameter87
            REAL, ALLOCATABLE,  DIMENSION(:,:,:) :: Parameter88
            REAL, ALLOCATABLE,  DIMENSION(:) :: Pooling160_Output_0_reshape0_shape
            REAL, ALLOCATABLE,  DIMENSION(:,:,:,:) :: Parameter193
            REAL, ALLOCATABLE,  DIMENSION(:) :: Parameter193_reshape1_shape
            REAL, ALLOCATABLE,  DIMENSION(:,:) :: Parameter194
            REAL, ALLOCATABLE,  DIMENSION(:,:) :: output0
            REAL, ALLOCATABLE,  DIMENSION(:,:) :: output1
            REAL, ALLOCATABLE, INTENT(OUT), DIMENSION(:,:) :: Plus214_Output_0
            REAL :: T1, T2
            CALL CPU_TIME(T1)
            
            !========Reshape============
            reshapeLayers(1)%reshape4d = RESHAPE(reshapeLayers(1)%reshape4d,(/SIZE(reshapeLayers(1)%reshape4d, dim = 4),&
                & SIZE(reshapeLayers(1)%reshape4d, dim = 3), SIZE(reshapeLayers(1)%reshape4d, dim = 2),&
                & SIZE(reshapeLayers(1)%reshape4d, dim = 1)/), order = [4, 3, 2, 1])
            output0 = RESHAPE(reshapeLayers(1)%reshape4d,(/256, 10/), order = [2, 1])

            !========Conv Layer============
            CALL conv(Input3, convLayers(1)%weights, convLayers(1)%biases,         (/1, 1/),         (/2, 2, 2, 2/),         (/1,&
                & 1/))
            !===========Add============
            Input3 = Input3 + RESHAPE(broadc(addLayers(1)%adder,        (/1, 8, 28, 28/),RESHAPE(        (/3, 28, 4, 28/),       &
                & (/2, 2/), order=[2,1])),         (/1, 8, 28, 28/))

            Input3 = relu4d(Input3)

            !========MaxPool Layer============
            CALL max_pool(Input3,maxpoolLayers(1), 0,         (/0, 0, 0, 0/),         (/2, 2/))

            !========Conv Layer============
            CALL conv(Input3, convLayers(2)%weights, convLayers(2)%biases,         (/1, 1/),         (/2, 2, 2, 2/),         (/1,&
                & 1/))
            !===========Add============
            Input3 = Input3 + RESHAPE(broadc(addLayers(2)%adder,        (/1, 16, 14, 14/),RESHAPE(        (/3, 14, 4, 14/),       &
                & (/2, 2/), order=[2,1])),         (/1, 16, 14, 14/))

            Input3 = relu4d(Input3)

            !========MaxPool Layer============
            CALL max_pool(Input3,maxpoolLayers(2), 0,         (/0, 0, 0, 0/),         (/3, 3/))

            !========Reshape============
            Input3 = RESHAPE(Input3,(/SIZE(Input3, dim = 4), SIZE(Input3, dim = 3), SIZE(Input3, dim = 2), SIZE(Input3, dim =&
                & 1)/), order = [4, 3, 2, 1])
            output1 = RESHAPE(Input3,(/1, 256/), order = [2, 1])

            !=======MatMul=========
            CALL matmul2D(output1, output0)
            
            !===========Add============
            output1 = output1 + RESHAPE(addLayers(3)%adder,         (/1, 10/))

            call CPU_TIME(T2)
            print *, "-------------"
            Plus214_Output_0 = output1
            print *, "TIME TAKEN:", T2-T1
>>>>>>> 932293133341125e44857a018a79d106ec53632e
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================
