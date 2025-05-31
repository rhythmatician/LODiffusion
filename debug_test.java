import fixtures.TestWorldFixtures;
import java.io.File;

public class debug_test {
    public static void main(String[] args) {
        System.out.println("Example world available: " + TestWorldFixtures.isExampleWorldAvailable());
        File[] files = TestWorldFixtures.getExampleWorldRegionFiles();
        System.out.println("Region files found: " + files.length);
        for (File file : files) {
            System.out.println(" - " + file.getPath());
        }
        
        System.out.println("Working directory: " + System.getProperty("user.dir"));
    }
}
