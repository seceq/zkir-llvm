// Simple test program for ZKIR LLVM backend
// Compile with: clang -O2 -emit-llvm -c simple.c -o simple.bc
// Then run: zkir-llvm simple.bc -o simple.zkir

int add(int a, int b) {
    return a + b;
}

int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int sum_array(int* arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    int x = add(10, 20);
    int y = factorial(5);

    int arr[5] = {1, 2, 3, 4, 5};
    int z = sum_array(arr, 5);

    return x + y + z;
}
