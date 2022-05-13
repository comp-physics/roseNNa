PROGRAM linear
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
    
    REAL :: inp1(1,5,1)
    REAL, ALLOCATABLE :: inp3(:,:,:)
    REAL, ALLOCATABLE :: hid1(:,:)
    REAL, ALLOCATABLE :: cell1(:,:)
    REAL, ALLOCATABLE :: out(:,:)
    REAL, ALLOCATABLE :: cellout(:,:)
    REAL :: inp2(2,3,3)
    REAL, ALLOCATABLE :: out2(:,:,:)
    ! REAL, ALLOCATABLE :: out2(:)
    ! REAL, ALLOCATABLE :: cellout2(:)
    ! INTEGER :: i
    REAL :: T1, T2
    ALLOCATE(hid1(2,1))
    ALLOCATE(cell1(2,1))
    inp2 = reshape ((/ 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0 , 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0/), shape(inp2))
    hid1 = reshape ((/ 1.0 , 1.0 /), shape(hid1))
    cell1 = reshape ((/ 1.0 , 1.0/), shape(cell1))
    ALLOCATE(inp3(2,6,6))
    inp3 = RESHAPE((/0.1535, 0.7006, 0.4696, 0.2404, 0.4496, 0.4557, 0.9583, 0.5196, 0.5715, &
    0.0655, 0.2453, 0.5083, 0.5991, 0.5690, 0.1250, 0.6286, 0.5039, 0.5927, &
    0.4749, 0.2383, 0.3091, 0.6463, 0.7774, 0.6773, 0.0038, 0.5568, 0.8577, &
    0.6815, 0.7714, 0.6924, 0.9020, 0.1982, 0.8751, 0.3438, 0.0171, 0.5159, &
    0.8374, 0.7789, 0.1574, 0.3831, 0.4589, 0.1489, 0.9174, 0.0732, 0.8255, &
    0.7209, 0.0538, 0.3203, 0.8771, 0.3480, 0.3369, 0.6313, 0.2190, 0.8578, &
    0.1807, 0.5276, 0.7660, 0.6342, 0.3823, 0.5833, 0.4597, 0.1423, 0.4167, &
    0.8521, 0.7152, 0.1629, 0.9007, 0.0354, 0.3485, 0.2165, 0.3287, 0.7753/), SHAPE(inp3)) !==testing purposes
    ! out = inp1
    CALL initialize()
    CALL CPU_TIME(T1)
    ! print *, lstmLayers(1)%whh(1,1)
    ! print *, lstmLayers(1)%whh(1,2)


    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
    CALL max_pool(inp3,maxpoolLayers(1))
    ! CALL conv(inp2, convLayers(1)%weights, convLayers(1)%biases, 2, 3, 3, 1)
    ! CALL lstm(inp1, hid1, cell1, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh, out, cellout)
    ! CALL linear_layer(out, linLayers(1))
    ! CALL linear_layer(out, linLayers(2))
    ! CALL linear_layer(out, linLayers(3))
    ! CALL linear_layer(out, linLayers(4))

    ! out = linear_layer(out, linLayers(4)%weights, linLayers(4)%biases)
    ! out = linLayers(4)%fn_ptr(out)
    !===================================================================================================================================================
    call CPU_TIME(T2)
    print *, "-------------"
    print *, inp3(1,:,1)
    print *, SHAPE(inp3)
    print *, architecture
    ! print *, out2(2,1,1)
    print *, "TIME TAKEN:", T2-T1    

END PROGRAM linear

!=======================================================================================
